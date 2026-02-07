/**
 * Implements the knock-on collision operator for binary large-angle collisions.
 */
#include "DREAM/Equations/Kinetic/KnockOnOperatorGeneral.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "DREAM/DREAMException.hpp"
#include "DREAM/Equations/KnockOnUtilities.hpp"
#include "DREAM/Settings/OptionConstants.hpp"

using namespace DREAM;

KnockOnOperatorGeneral::KnockOnOperatorGeneral(
    FVM::Grid *gridKnockon, FVM::Grid *grid_primary, FVM::UnknownQuantityHandler *unknowns,
    len_t id_f_primary, real_t pCutoff, real_t scaleFactor, len_t n_xi_stars_tabulate,
    len_t n_points_integral
)
    : FVM::EquationTerm(gridKnockon),
      gridPrimary(grid_primary),
      unknowns(unknowns),
      id_f_primary(id_f_primary),
      pCutoff(pCutoff),
      scaleFactor(scaleFactor),
      nXiStarsTabulate(n_xi_stars_tabulate),
      nPointsIntegral(n_points_integral) {
    SetName("KnockOnOperatorGeneral");

    ValidateInputParameters();
    ValidateGridAssumptions();

    id_ntot = unknowns->GetUnknownID(OptionConstants::UQTY_N_TOT);
    AddUnknownForJacobian(unknowns, id_ntot);

    t_source_rebuilt = -std::numeric_limits<real_t>::infinity();
    AllocateAndBuildTables();
}

KnockOnOperatorGeneral::~KnockOnOperatorGeneral() { Deallocate(); }

void KnockOnOperatorGeneral::ValidateInputParameters() const {
    if (unknowns == nullptr) {
        throw DREAMException("KnockOnOperatorGeneral: 'unknowns' must not be null.");
    }
    if (!(pCutoff > 0)) {
        throw DREAMException(
            "KnockOnOperatorGeneral: invalid pCutoff=%.16g (must be > 0).", pCutoff
        );
    }
    if (nXiStarsTabulate < 2) {
        throw DREAMException(
            "KnockOnOperatorGeneral: n_xi_stars_tabulate must be >= 2 (got %ld).",
            (long)nXiStarsTabulate
        );
    }
    if (nPointsIntegral < 1) {
        throw DREAMException(
            "KnockOnOperatorGeneral: n_points_integral must be >= 1 (got %ld).",
            (long)nPointsIntegral
        );
    }
}

// Ensure that the assumptions we make on the grid are justified.
void KnockOnOperatorGeneral::ValidateGridAssumptions() const {
    if (grid == nullptr || gridPrimary == nullptr)
        throw DREAMException("KnockOnOperatorGeneral: grid pointers must not be null.");

    // Require same Nr.
    if (grid->GetNr() != gridPrimary->GetNr())
        throw DREAMException("KnockOnOperatorGeneral: grid and grid_primary must have same Nr.");

    // Require p-\xi grids and uniform momentum resolution across radii.
    // KnockOnUtilities assumes the momentum grids are identical at all radii.
    for (len_t ir = 1; ir < grid->GetNr(); ++ir) {
        if (grid->GetNp1(ir) != grid->GetNp1(0) || grid->GetNp2(ir) != grid->GetNp2(0)) {
            throw DREAMException(
                "KnockOnOperatorGeneral: requires uniform knock-on Np across radii."
            );
        }

        if (gridPrimary->GetNp1(ir) != gridPrimary->GetNp1(0) ||
            gridPrimary->GetNp2(ir) != gridPrimary->GetNp2(0)) {
            throw DREAMException("KnockOnOperatorGeneral: requires uniform primary Np across radii."
            );
        }
    }
}

void KnockOnOperatorGeneral::Deallocate() {
    const len_t Nr = (grid != nullptr) ? grid->GetNr() : 0;

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
        for (len_t ir = 0; ir < Nr; ++ir) {
            delete[] xiInterp[ir];
        }
        delete[] xiInterp;
        xiInterp = nullptr;
    }

    if (xiStarsTab != nullptr) {
        delete[] xiStarsTab;
        xiStarsTab = nullptr;
    }

    if (mollerSMatrix != nullptr) {
        delete[] mollerSMatrix;
        mollerSMatrix = nullptr;
    }

    if (sourceVector != nullptr) {
        delete[] sourceVector;
        sourceVector = nullptr;
    }

    if (primaryWeights != nullptr) {
        delete[] primaryWeights;
        primaryWeights = nullptr;
    }

    if (pitchAccum != nullptr) {
        delete[] pitchAccum;
        pitchAccum = nullptr;
    }
}

void KnockOnOperatorGeneral::AllocateScratchBuffers() {
    // We allocate scratch buffers based on radius 0, relying on validated uniform
    // momentum resolution across radii.
    const auto *mgK = grid->GetMomentumGrid(0);
    const auto *mgP = gridPrimary->GetMomentumGrid(0);

    const len_t Np1P = mgP->GetNp1();
    const len_t NxiP = mgP->GetNp2();
    const len_t NxiK = mgK->GetNp2();

    primaryWeights = new real_t[Np1P * NxiP];
    pitchAccum = new real_t[NxiK];
}

void KnockOnOperatorGeneral::AllocateAndBuildTables() {
    const len_t Nr = grid->GetNr();

    // Delta-kernel tabulation.
    deltaTable = new real_t **[Nr];
    deltaTableStorage = new real_t *[Nr];

    for (len_t ir = 0; ir < Nr; ++ir) {
        const len_t NxiK = grid->GetNp2(ir);
        const len_t NxiP = gridPrimary->GetNp2(ir);
        const len_t planeSize = NxiK * NxiP;

        deltaTable[ir] = new real_t *[nXiStarsTabulate];
        deltaTableStorage[ir] = new real_t[nXiStarsTabulate * planeSize];

        for (len_t m = 0; m < nXiStarsTabulate; ++m) {
            deltaTable[ir][m] = deltaTableStorage[ir] + m * planeSize;
        }
    }

    // Momentum-space kernel.
    mollerSMatrix = new real_t[grid->GetNp1(0) * gridPrimary->GetNp1(0)];

    // Full source vector.
    sourceVector = new real_t[grid->GetNCells()];

    // Construct xi_star tabulation grid. We use the maximum primary momentum to
    // determine the allowed knock-on domain.
    const auto *mgK = grid->GetMomentumGrid(0);
    const auto *mgP = gridPrimary->GetMomentumGrid(0);

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
    // xi_star interpolation metadata.
    xiInterp = new XiStarInterp *[Nr];
    for (len_t ir = 0; ir < Nr; ++ir) {
        xiInterp[ir] = new XiStarInterp[grid->GetNp1(ir) * gridPrimary->GetNp1(ir)];
    }
    AllocateScratchBuffers();
    BuildMollerSMatrix();
    TabulateDeltaMatrixOnXiStarGrid();
    BuildXiStarInterp();
}

void KnockOnOperatorGeneral::BuildMollerSMatrix() {
    const len_t NpK = grid->GetNp1(0);
    const len_t NpP = gridPrimary->GetNp1(0);

    for (len_t i = 0; i < NpK; ++i) {
        for (len_t k = 0; k < NpP; ++k) {
            mollerSMatrix[i * NpP + k] = KnockOnUtilities::EvaluateMollerFluxMatrixElementOnGrid(
                i, k, grid, gridPrimary, pCutoff
            );
        }
    }
}

void KnockOnOperatorGeneral::BuildXiStarInterp() {
    const real_t inv_dXiStar = 1.0 / dXiStar;

    for (len_t ir = 0; ir < grid->GetNr(); ++ir) {
        const auto *mgK = grid->GetMomentumGrid(ir);
        const auto *mgP = gridPrimary->GetMomentumGrid(ir);

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

                const real_t s = (xs - xiStarMin) * inv_dXiStar;  // >= 0
                len_t m0 = (len_t)s;
                if (m0 >= nXiStarsTabulate - 1) {
                    m0 = nXiStarsTabulate - 2;
                }
                T.clamp = Interp;
                T.m0 = m0;
                T.w1 = s - (real_t)m0;  // in [0,1)
            }
        }
    }
}

/**
 * Build per-(k,l) primary pitch weights W_l from f_primary at radius ir.
 * Writes W as a (Np1P x NxiP) table, used by AccumulateAngleKernel().
 * Input pointer f_primary_ir must point to the first cell of the primary grid at this radius.
 */
void KnockOnOperatorGeneral::BuildPrimaryWeights(
    len_t ir, const real_t *f_primary_ir, real_t *W_k_l
) const {
    const auto *mgP = gridPrimary->GetMomentumGrid(ir);

    const len_t Np1P = mgP->GetNp1();
    const len_t NxiP = mgP->GetNp2();

    const real_t *dp1 = mgP->GetDp1();
    const real_t *dxi1 = mgP->GetDp2();
    const real_t *VpP = gridPrimary->GetVp(ir);

    // f_primary_ir is (l,k) contiguous: idx = l*Np1P + k
    for (len_t l = 0; l < NxiP; ++l) {
        const real_t dxi = dxi1[l];
        const len_t base_lk = l * Np1P;
        for (len_t k = 0; k < Np1P; ++k) {
            const len_t idx_lk = base_lk + k;
            W_k_l[k * NxiP + l] = dp1[k] * dxi * VpP[idx_lk] * f_primary_ir[idx_lk];
        }
    }
}

/**
 * Choose delta-table plane(s) in xi* for (ir,i,k).
 * Returns either one clamped plane (D1==nullptr) or two neighboring planes with
 * linear weights (w0,w1).
 */
void KnockOnOperatorGeneral::SelectDeltaPlanes(
    len_t ir, len_t i, len_t k, const real_t *&D0, const real_t *&D1, real_t &w0, real_t &w1
) const {
    const auto *mgP = gridPrimary->GetMomentumGrid(ir);
    const len_t Np1P = mgP->GetNp1();

    const XiStarInterp &T = xiInterp[ir][i * Np1P + k];

    D0 = nullptr;
    D1 = nullptr;
    w0 = 1.0;
    w1 = 0.0;

    switch (T.clamp) {
        case XiClamp::ClampLow:
            D0 = deltaTable[ir][0];
            return;
        case XiClamp::ClampHigh:
            D0 = deltaTable[ir][nXiStarsTabulate - 1];
            return;
        case XiClamp::Interp: {
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
 * Add pitch-angle redistribution for one (ir,i,k) pair into outPitch_j.
 * Computes: outPitch_j[j] += Sik * Σ_l W_l[l] * Δ_{j l}(xi*(p_i,p1_k)).
 * outPitch_j is updated in-place (not cleared here).
 */
void KnockOnOperatorGeneral::AccumulateAngleKernel(
    len_t ir, len_t i, len_t k, const real_t *W_l, real_t Sik, real_t *outPitch_j
) const {
    if (Sik == 0) {
        return;
    }
    const auto *mgK = grid->GetMomentumGrid(ir);
    const auto *mgP = gridPrimary->GetMomentumGrid(ir);

    const len_t NxiK = mgK->GetNp2();
    const len_t NxiP = mgP->GetNp2();

    const real_t *D0, *D1;
    real_t w0, w1;
    SelectDeltaPlanes(ir, i, k, D0, D1, w0, w1);

    // Apply columns: out[j] += Sik * Σ_l W_l[l] * Δ_{j l}
    if (D1 == nullptr) {
        for (len_t l = 0; l < NxiP; ++l) {
            const real_t W = W_l[l];
            if (W == 0) {
                continue;
            }
            const real_t scale = Sik * W;
            const real_t *col0 = D0 + l * NxiK;
            for (len_t j = 0; j < NxiK; ++j) {
                outPitch_j[j] += scale * col0[j];
            }
        }
    } else {
        for (len_t l = 0; l < NxiP; ++l) {
            const real_t W = W_l[l];
            if (W == 0) {
                continue;
            }
            const real_t scale = Sik * W;
            const real_t *col0 = D0 + l * NxiK;
            const real_t *col1 = D1 + l * NxiK;
            for (len_t j = 0; j < NxiK; ++j) {
                outPitch_j[j] += scale * (w0 * col0[j] + w1 * col1[j]);
            }
        }
    }
}

/**
 * Build the knock-on source vector from the primary distribution f_primary.
 * Fills sourceVector on the knock-on grid by integrating over primary (k,l),
 * applying the Møller kernel, and dividing by local phase-space volume Vp.
 */
void KnockOnOperatorGeneral::SetSourceVector(const real_t *f_primary) {
    len_t offK = 0;
    len_t offP = 0;

    const len_t Nr = grid->GetNr();

    if (primaryWeights == nullptr || pitchAccum == nullptr)
        throw DREAMException("KnockOnOperatorGeneral: internal scratch buffers not allocated.");

    for (len_t ir = 0; ir < Nr; ++ir) {
        const auto *mgK = grid->GetMomentumGrid(ir);
        const auto *mgP = gridPrimary->GetMomentumGrid(ir);

        const len_t Np1K = mgK->GetNp1();
        const len_t NxiK = mgK->GetNp2();
        const len_t Np1P = mgP->GetNp1();
        const len_t NxiP = mgP->GetNp2();

        const real_t *VpK = grid->GetVp(ir);

        BuildPrimaryWeights(ir, f_primary + offP, primaryWeights);

        // For each outgoing momentum cell i: build pitch distribution by summing over k.
        for (len_t i = 0; i < Np1K; ++i) {
            for (len_t j = 0; j < NxiK; ++j) {
                pitchAccum[j] = 0.0;
            }
            for (len_t k = 0; k < Np1P; ++k) {
                const real_t Sik = MollerDifferentialCrossSection(ir, i, k);
                const real_t *W_l = primaryWeights + k * NxiP;
                AccumulateAngleKernel(ir, i, k, W_l, Sik, pitchAccum);
            }

            // Write to sourceVector at this i.
            for (len_t j = 0; j < NxiK; ++j) {
                const len_t idxK = offK + j * Np1K + i;
                const real_t Vp = VpK[j * Np1K + i];
                sourceVector[idxK] = (Vp != 0) ? (pitchAccum[j] * (scaleFactor / Vp)) : 0.0;
            }
        }

        offK += mgK->GetNCells();
        offP += mgP->GetNCells();
    }
}

void KnockOnOperatorGeneral::Rebuild(real_t t, real_t /*dt*/, FVM::UnknownQuantityHandler *uqh) {
    // Update unknown handler pointer if the caller provided one.
    if (uqh != nullptr) {
        this->unknowns = uqh;
    }

    // avoid rebuild if time hasn't changed (more than roundoff ~eps)
    constexpr real_t TIME_EPS_FACTOR = 100;
    const bool timeHasUpdated =
        std::abs(t - t_source_rebuilt) > TIME_EPS_FACTOR * std::numeric_limits<real_t>::epsilon();

    if (timeHasUpdated) {
        // Handle source term explicitly: use value at previous time-step.
        const real_t *f_primary = unknowns->GetUnknownDataPrevious(id_f_primary);
        SetSourceVector(f_primary);
        t_source_rebuilt = t;
    }
}

// Here, x is assumed to be the density to which the source is proportional (n_tot).
void KnockOnOperatorGeneral::SetVectorElements(real_t *vec, const real_t *x) {
    len_t offset = 0;
    for (len_t ir = 0; ir < grid->GetNr(); ++ir) {
        const auto *mg = grid->GetMomentumGrid(ir);
        const len_t Np = mg->GetNp1();
        const len_t Nxi = mg->GetNp2();

        for (len_t j = 0; j < Nxi; ++j) {
            const len_t base = offset + Np * j;
            for (len_t i = 0; i < Np; ++i) {
                const len_t ind = base + i;
                vec[ind] += x[ir] * sourceVector[ind];
            }
        }
        offset += mg->GetNCells();
    }
}

bool KnockOnOperatorGeneral::SetJacobianBlock(
    const len_t /*uqtyId*/, const len_t derivId, FVM::Matrix *jac, const real_t * /*x*/
) {
    if (derivId != id_ntot) {
        return false;
    }
    len_t offset = 0;
    for (len_t ir = 0; ir < grid->GetNr(); ++ir) {
        const auto *mg = grid->GetMomentumGrid(ir);
        const len_t Np = mg->GetNp1();
        const len_t Nxi = mg->GetNp2();

        for (len_t j = 0; j < Nxi; ++j) {
            const len_t base = offset + Np * j;
            for (len_t i = 0; i < Np; ++i) {
                const len_t ind = base + i;
                jac->SetElement(ind, ir, sourceVector[ind]);
            }
        }

        offset += mg->GetNCells();
    }

    return true;
}

bool KnockOnOperatorGeneral::GridRebuilt() {
    ValidateGridAssumptions();
    Deallocate();
    t_source_rebuilt = -std::numeric_limits<real_t>::infinity();
    AllocateAndBuildTables();
    return true;
}

// Tabulate the pitch-angle Delta kernel on the reference xi_star grid.
void KnockOnOperatorGeneral::TabulateDeltaMatrixOnXiStarGrid() {
    const len_t Nr = grid->GetNr();
    for (len_t ir = 0; ir < Nr; ++ir) {
        const len_t NxiK = grid->GetNp2(ir);
        const len_t NxiP = gridPrimary->GetNp2(ir);

        for (len_t m = 0; m < nXiStarsTabulate; ++m) {
            real_t *plane = deltaTable[ir][m];
            for (len_t l = 0; l < NxiP; ++l) {
                const len_t offset = l * NxiK;
                KnockOnUtilities::SetDeltaMatrixColumnOnGrid(
                    ir, xiStarsTab[m], l, grid, gridPrimary, plane + offset, nPointsIntegral
                );
            }
        }
    }
}
