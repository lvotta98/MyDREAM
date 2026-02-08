#include "DREAM/Equations/Kinetic/MollerDeltaAngleKernel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "DREAM/DREAMException.hpp"
#include "DREAM/Equations/KnockOnUtilities.hpp"

using namespace DREAM;

MollerDeltaAngleKernel::MollerDeltaAngleKernel(
    const FVM::Grid *grid_knockon, const FVM::Grid *grid_primary, real_t p_cutoff,
    len_t n_xi_stars_tabulate, len_t n_points_integral
)
    : gridK(grid_knockon),
      gridP(grid_primary),
      pCutoff(p_cutoff),
      nXiStarsTabulate(n_xi_stars_tabulate),
      nPointsIntegral(n_points_integral) {
    ValidateInputParameters();
    GridRebuilt();
}

MollerDeltaAngleKernel::~MollerDeltaAngleKernel() { Deallocate(); }

void MollerDeltaAngleKernel::Deallocate() {
    const len_t Nr = (gridK != nullptr) ? gridK->GetNr() : 0;

    if (deltaTable != nullptr) {
        for (len_t ir = 0; ir < Nr; ++ir) {
            delete[] deltaTableStorage[ir];
            delete[] deltaTable[ir];
        }
        delete[] deltaTableStorage;
        delete[] deltaTable;
        deltaTableStorage = nullptr;
        deltaTable = nullptr;
    }

    if (xiInterp != nullptr) {
        for (len_t ir = 0; ir < Nr; ++ir) delete[] xiInterp[ir];
        delete[] xiInterp;
        xiInterp = nullptr;
    }

    if (xiStarsTab != nullptr) {
        delete[] xiStarsTab;
        xiStarsTab = nullptr;
    }
}

void DREAM::MollerDeltaAngleKernel::ValidateGridAssumptions() const {
    if (gridK == nullptr || gridP == nullptr)
        throw DREAMException("MollerDeltaAngleKernel: grid pointers must not be null.");

    if (!(pCutoff > 0))
        throw DREAMException(
            "MollerDeltaAngleKernel: invalid pCutoff=%.16g (must be > 0).", pCutoff
        );

    if (nXiStarsTabulate < 2)
        throw DREAMException(
            "MollerDeltaAngleKernel: nXiStarsTabulate must be >= 2 (got %ld).",
            (long)nXiStarsTabulate
        );
    if (nPointsIntegral < 1)
        throw DREAMException(
            "MollerDeltaAngleKernel: nPointsIntegral must be >= 1 (got %ld).", (long)nPointsIntegral
        );

    const len_t NrK = gridK->GetNr();
    const len_t NrP = gridP->GetNr();
    if (NrK != NrP)
        throw DREAMException("MollerDeltaAngleKernel: gridK and gridP must have same Nr.");

    for (len_t ir = 1; ir < NrK; ++ir) {
        if (gridK->GetNp1(ir) != gridK->GetNp1(0) || gridK->GetNp2(ir) != gridK->GetNp2(0)) {
            throw DREAMException(
                "MollerDeltaAngleKernel: requires uniform knock-on momentum grid across radii."
            );
        }
        if (gridP->GetNp1(ir) != gridP->GetNp1(0) || gridP->GetNp2(ir) != gridP->GetNp2(0)) {
            throw DREAMException(
                "MollerDeltaAngleKernel: requires uniform primary momentum grid across radii."
            );
        }
    }
}

void DREAM::MollerDeltaAngleKernel::ValidateInputParameters() const {
    if (nXiStarsTabulate < 2) {
        throw DREAMException(
            "MollerDeltaAngleKernel: nXiStarsTabulate must be >= 2 (got %ld).",
            (long)nXiStarsTabulate
        );
    }
    if (nPointsIntegral < 1) {
        throw DREAMException(
            "MollerDeltaAngleKernel: nPointsIntegral must be >= 1 (got %ld).", (long)nPointsIntegral
        );
    }
}

// Generate a xi_star grid on which we tabulate the kinematic delta. 
// To get maximum bang for the back we identify its min and max values
// on the given grids and only sample between those.
void MollerDeltaAngleKernel::BuildXiStarTabulationGrid() {
    const auto *mgK = gridK->GetMomentumGrid(0);
    const auto *mgP = gridP->GetMomentumGrid(0);

    const real_t p1_max = mgP->GetP1(mgP->GetNp1() - 1);
    const real_t g1_max = std::sqrt(1 + p1_max * p1_max);
    const real_t g_max = (g1_max + 1) / 2;
    const real_t p_max = std::sqrt(g_max * g_max - 1);

    const real_t p_min = std::max(pCutoff, mgK->GetP1(0));

    xiStarMin = KnockOnUtilities::Kinematics::EvaluateXiStar(p_min, p1_max);
    xiStarMax = KnockOnUtilities::Kinematics::EvaluateXiStar(p_max, p1_max);
    dXiStar = (xiStarMax - xiStarMin) / (real_t)(nXiStarsTabulate - 1);

    xiStarsTab = new real_t[nXiStarsTabulate];
    for (len_t i = 0; i < nXiStarsTabulate; ++i) {
        xiStarsTab[i] = xiStarMin + (real_t)i * dXiStar;
    }
}

void MollerDeltaAngleKernel::AllocateDeltaTables() {
    const len_t Nr = gridK->GetNr();

    deltaTable = new real_t **[Nr];
    deltaTableStorage = new real_t *[Nr];

    for (len_t ir = 0; ir < Nr; ++ir) {
        const len_t NxiK = gridK->GetNp2(ir);
        const len_t NxiP = gridP->GetNp2(ir);
        const len_t planeSize = NxiK * NxiP;

        deltaTable[ir] = new real_t *[nXiStarsTabulate];
        deltaTableStorage[ir] = new real_t[nXiStarsTabulate * planeSize];

        for (len_t m = 0; m < nXiStarsTabulate; ++m) {
            deltaTable[ir][m] = deltaTableStorage[ir] + m * planeSize;
        }
    }

    xiInterp = new XiStarInterp *[Nr];
    for (len_t ir = 0; ir < Nr; ++ir) {
        xiInterp[ir] = new XiStarInterp[gridK->GetNp1(ir) * gridP->GetNp1(ir)];
    }
}

// The real kinematic delta depends on p_i, p1_k only indirectly via xi_star(p, p1),
// therefore we here tabulate on a grid of xi_star values.
void MollerDeltaAngleKernel::TabulateDeltaMatrixOnXiStarGrid() {
    const len_t Nr = gridK->GetNr();
    for (len_t ir = 0; ir < Nr; ++ir) {
        const len_t NxiK = gridK->GetNp2(ir);
        const len_t NxiP = gridP->GetNp2(ir);

        for (len_t m = 0; m < nXiStarsTabulate; ++m) {
            real_t *plane = deltaTable[ir][m];
            for (len_t l = 0; l < NxiP; ++l) {
                const len_t offset = l * NxiK;
                KnockOnUtilities::SetDeltaMatrixColumnOnGrid(
                    ir, xiStarsTab[m], l, gridK, gridP, plane + offset, nPointsIntegral
                );
            }
        }
    }
}


// Precompute interpolation tables for linear interpolation in delta vs xi_star.
void MollerDeltaAngleKernel::BuildXiStarInterp() {
    const real_t inv_dXiStar = 1.0 / dXiStar;

    for (len_t ir = 0; ir < gridK->GetNr(); ++ir) {
        const auto *mgK = gridK->GetMomentumGrid(ir);
        const auto *mgP = gridP->GetMomentumGrid(ir);

        const len_t NpK = mgK->GetNp1();
        const len_t NpP = mgP->GetNp1();

        const real_t *p = mgK->GetP1();
        const real_t *p1 = mgP->GetP1();

        for (len_t i = 0; i < NpK; ++i) {
            for (len_t k = 0; k < NpP; ++k) {
                XiStarInterp &T = xiInterp[ir][i * NpP + k];
                const real_t xs = KnockOnUtilities::Kinematics::EvaluateXiStar(p[i], p1[k]);

                if (xs <= xiStarMin) {
                    T.clamp = ClampLow;
                    T.m0 = 0;
                    T.w1 = 0;
                    continue;
                }
                if (xs >= xiStarMax) {
                    T.clamp = ClampHigh;
                    T.m0 = 0;
                    T.w1 = 0;
                    continue;
                }

                const real_t s = (xs - xiStarMin) * inv_dXiStar;
                len_t m0 = (len_t)s;
                if (m0 >= nXiStarsTabulate - 1) {
                    m0 = nXiStarsTabulate - 2;
                }
                T.clamp = Interp;
                T.m0 = m0;
                T.w1 = s - (real_t)m0;
            }
        }
    }
}

void MollerDeltaAngleKernel::GridRebuilt() {
    Deallocate();
    ValidateGridAssumptions();

    BuildXiStarTabulationGrid();
    AllocateDeltaTables();
    TabulateDeltaMatrixOnXiStarGrid();
    BuildXiStarInterp();
}

/**
 * Select the tabulated delta-kernel plane(s) corresponding to xi_star(p_i,p1_k).
 *
 * Key idea:
 *  - If xi_star lies outside the tabulated range, clamp to the nearest endpoint plane.
 *  - Otherwise return the two neighboring planes and linear interpolation weights.
 *
 * Output convention:
 *  - Returns either one plane (D1==nullptr) or two planes (D0,D1) with weights (w0,w1).
 */
void MollerDeltaAngleKernel::GetDeltaInterpRaw(
    len_t ir, len_t i, len_t k, const real_t *&D0, const real_t *&D1, real_t &w0, real_t &w1
) const {
    const auto *mgP = gridP->GetMomentumGrid(ir);
    const len_t Np1P = mgP->GetNp1();

    const XiStarInterp &T = xiInterp[ir][i * Np1P + k];

    D0 = nullptr;
    D1 = nullptr;
    w0 = 1.0;
    w1 = 0.0;

    switch (T.clamp) {
        case ClampLow:
            D0 = deltaTable[ir][0];
            return;
        case ClampHigh:
            D0 = deltaTable[ir][nXiStarsTabulate - 1];
            return;
        case Interp:
        default: {
            const len_t m0 = T.m0;
            D0 = deltaTable[ir][m0];
            D1 = deltaTable[ir][m0 + 1];
            w1 = T.w1;
            w0 = 1.0 - w1;
            return;
        }
    }
}

/**
 * Return a normalized delta-kernel interpolation descriptor for xi_star(p_i,p1_k).
 *
 * Key idea:
 *  - Wrap GetDeltaInterpRaw() but normalize the representation so interpolation is always
 *    "two-plane" form: D1 is never null (clamped case is encoded as D1=D0, w1=0).
 *
 * This removes branching in the hot accumulation loop.
 */
MollerDeltaAngleKernel::DeltaInterp MollerDeltaAngleKernel::GetDeltaInterp(
    len_t ir, len_t i, len_t k
) const {
    const real_t *D0 = nullptr, *D1 = nullptr;
    real_t w0 = 1.0, w1 = 0.0;

    GetDeltaInterpRaw(ir, i, k, D0, D1, w0, w1);

    // Normalize: represent “single plane” as interpolation with D1=D0, w1=0.
    if (D1 == nullptr) {
        D1 = D0;
        w0 = 1.0;
        w1 = 0.0;
    }

    return {D0, D1, w0, w1};
}

/**
 * Accumulate the pitch-angle redistribution contribution for fixed (ir,i,k).
 *
 * Computes:
 *   C_j += sigma_ik * Σ_l W_{k l} * Delta_{j l}(xi*(p_i,p1_k)),
 * where W_{k l} already includes dp1_k * dxi_l * Vp_{k l} * f_{k l}.
 *
 * Key idea:
 *  - The delta kernel is tabulated on a xi_star grid, and here we interpolate
 *    linearly in that table and apply to the distribution.
 */
void MollerDeltaAngleKernel::AccumulatePitch(
    len_t ir, len_t i, len_t k, const real_t *Wkl, real_t sigma_ik, real_t *Cj
) const {
    if (sigma_ik == 0) return;

    const auto *mgK = gridK->GetMomentumGrid(ir);
    const auto *mgP = gridP->GetMomentumGrid(ir);

    const len_t NxiK = mgK->GetNp2();
    const len_t NxiP = mgP->GetNp2();

    const DeltaInterp I = GetDeltaInterp(ir, i, k);

    // For each primary pitch cell l:
    //   C_j += (sigma_ik * Wkl[l]) * [ w0*Delta0(:,l) + w1*Delta1(:,l) ]
    for (len_t l = 0; l < NxiP; ++l) {
        const real_t W = Wkl[l];
        if (W == 0) {
            continue;
        }

        const real_t alpha = sigma_ik * W;

        const real_t *col0 = I.D0 + l * NxiK;
        const real_t *col1 = I.D1 + l * NxiK;

        for (len_t j = 0; j < NxiK; ++j) {
            Cj[j] += alpha * (I.w0 * col0[j] + I.w1 * col1[j]);
        }
    }
}
