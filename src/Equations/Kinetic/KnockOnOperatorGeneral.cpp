/**
 * Implements the knock-on collision operator for binary large-angle collisions.
 */
#include "DREAM/Equations/Kinetic/KnockOnOperatorGeneral.hpp"

#include <cmath>
#include <limits>

#include "DREAM/Equations/KnockOnUtilities.hpp"
#include "DREAM/Settings/OptionConstants.hpp"

using namespace DREAM;

KnockOnOperatorGeneral::KnockOnOperatorGeneral(
    FVM::Grid *grid_knockon, FVM::Grid *grid_primary, FVM::UnknownQuantityHandler *unknowns,
    len_t id_f_primary, real_t pCutoff, real_t scaleFactor, len_t n_xi_stars_tabulate,
    len_t n_points_integral
)
    : FVM::EquationTerm(grid_knockon),
      grid_primary(grid_primary),
      unknowns(unknowns),
      id_f_primary(id_f_primary),
      pCutoff(pCutoff),
      scaleFactor(scaleFactor),
      n_xi_stars_tabulate(n_xi_stars_tabulate),
      n_points_integral(n_points_integral) {
    SetName("KnockOnOperatorGeneral");
    id_ntot = unknowns->GetUnknownID(OptionConstants::UQTY_N_TOT);
    id_Efield = unknowns->GetUnknownID(OptionConstants::UQTY_E_FIELD);
    t_source_rebuilt = -std::numeric_limits<real_t>::infinity();
    AddUnknownForJacobian(unknowns, id_ntot);
    Allocate();
}

KnockOnOperatorGeneral::~KnockOnOperatorGeneral() { Deallocate(); }

void KnockOnOperatorGeneral::Deallocate() {
    len_t Nr = grid->GetNr();
    if (deltaTable != nullptr) {
        for (len_t ir = 0; ir < Nr; ir++) {
            // XXX: assume p-xi grid
            for (len_t m = 0; m < n_xi_stars_tabulate; m++) {
                delete[] deltaTable[ir][m];
            }
            delete[] deltaTable[ir];
        }
        delete[] deltaTable;
    }
    if (xiInterp != nullptr) {
        for (len_t ir = 0; ir < Nr; ir++) {
            delete[] xiInterp[ir];
        }
        delete[] xiInterp;
    }
    if (xiStarsTab != nullptr) {
        delete[] xiStarsTab;
    }
    if (mollerSMatrix != nullptr) {
        delete[] mollerSMatrix;
    }
    if (sourceVector != nullptr) {
        delete[] sourceVector;
    }
}

void KnockOnOperatorGeneral::Allocate() {
    len_t Nr = grid->GetNr();
    // We introduce deltaTable such that
    // deltaTable[ir][m][j + l*Nxi] = Delta_{j l}(xi_star[m])
    deltaTable = new real_t **[Nr];
    for (len_t ir = 0; ir < Nr; ir++) {
        deltaTable[ir] = new real_t *[n_xi_stars_tabulate];
        for (len_t m = 0; m < n_xi_stars_tabulate; m++) {
            deltaTable[ir][m] = new real_t[grid->GetNp2(ir) * grid_primary->GetNp2(ir)];
        }
    }

    mollerSMatrix = new real_t[grid->GetNp1(0) * grid_primary->GetNp1(0)];

    sourceVector = new real_t[grid->GetNCells()];

    // XXX: assume same momentum grid at all radii, and p-xi grid
    real_t p1_max = grid->GetMomentumGrid(0)->GetP1(grid_primary->GetNp1(0) - 1);
    real_t g1_max = sqrt(1 + p1_max * p1_max);
    real_t g_max = (g1_max + 1) / 2;
    real_t p_max = sqrt(g_max * g_max - 1);

    real_t p_min = std::max(pCutoff, grid->GetMomentumGrid(0)->GetP1(0));

    xiStar_min = KnockOnUtilities::Kinematics::EvaluateXiStar(p_min, p1_max);
    xiStar_max = KnockOnUtilities::Kinematics::EvaluateXiStar(p_max, p1_max);
    dXiStar = (xiStar_max - xiStar_min) / (real_t)(n_xi_stars_tabulate - 1);
    xiStarsTab = new real_t[n_xi_stars_tabulate];
    for (len_t i = 0; i < n_xi_stars_tabulate; i++) {
        xiStarsTab[i] = xiStar_min + (real_t)i * dXiStar;
    }

    xiInterp = new XiStarInterp *[Nr];
    for (len_t ir = 0; ir < Nr; ++ir) {
        xiInterp[ir] = new XiStarInterp[grid->GetNp1(ir) * grid_primary->GetNp1(ir)];
    }
    SetMollerSMatrix(mollerSMatrix);
    TabulateDeltaMatrixOnXiStarGrid();
    BuildXiStarInterp();
}

void KnockOnOperatorGeneral::BuildXiStarInterp() {
    const real_t inv_dXiStar = 1.0 / dXiStar;

    for (len_t ir = 0; ir < grid->GetNr(); ++ir) {
        auto *mgK = grid->GetMomentumGrid(ir);
        auto *mgP = grid_primary->GetMomentumGrid(ir);

        const len_t NpK = mgK->GetNp1();
        const len_t NpP = mgP->GetNp1();

        const real_t *p = mgK->GetP1();
        const real_t *p1 = mgP->GetP1();

        for (len_t i = 0; i < NpK; i++) {
            for (len_t k = 0; k < NpP; k++) {
                XiStarInterp &T = xiInterp[ir][i * NpP + k];

                real_t xs = KnockOnUtilities::Kinematics::EvaluateXiStar(p[i], p1[k]);

                if (xs <= xiStar_min) {
                    T.clamp = 1;
                    continue;
                }
                if (xs >= xiStar_max) {
                    T.clamp = 2;
                    continue;
                }

                real_t s = (xs - xiStar_min) * inv_dXiStar;  // >=0
                len_t m0 = (len_t)s;                         // trunc ok since s>=0

                if (m0 >= n_xi_stars_tabulate - 1) {
                    m0 = n_xi_stars_tabulate - 2;
                }

                T.clamp = 0;
                T.m0 = m0;
                T.w1 = s - (real_t)m0;  // in [0,1)
            }
        }
    }
}

void KnockOnOperatorGeneral::SetSourceVector(const real_t *f_primary) {
    len_t offK = 0;
    len_t offP = 0;

    const len_t Nr = grid->GetNr();

    for (len_t ir = 0; ir < Nr; ir++) {
        auto *mgK = grid->GetMomentumGrid(ir);
        auto *mgP = grid_primary->GetMomentumGrid(ir);

        const len_t Np1K = mgK->GetNp1();
        const len_t Np2K = mgK->GetNp2();
        const len_t Np1P = mgP->GetNp1();
        const len_t Np2P = mgP->GetNp2();

        const real_t *VpK = grid->GetVp(ir);
        const real_t *VpP = grid_primary->GetVp(ir);
        const real_t *VpOverP2 = grid->GetVpOverP2AtZero(ir);
        const real_t *dp1 = mgP->GetDp1();
        const real_t *dxi1 = mgP->GetDp2();

        // Precompute the part of the integrand that lives only on the primary grid
        std::vector<real_t> Wkl;
        Wkl.resize((size_t)Np1P * (size_t)Np2P);
        for (len_t l = 0; l < Np2P; l++) {
            const real_t dxi = dxi1[l];
            const len_t base = l * Np1P;
            for (len_t k = 0; k < Np1P; k++) {
                const len_t idxP = base + k;
                const real_t f = f_primary[offP + idxP];
                Wkl[idxP] = dp1[k] * dxi * VpP[idxP] * f;
            }
        }

        for (len_t j = 0; j < Np2K; j++) {
            if (VpOverP2[j] == 0) continue;

            const len_t baseK = offK + j * Np1K;

            for (len_t i = 0; i < Np1K; i++) {
                const len_t idxK = baseK + i;

                real_t sum = 0;

                // integrate over primary momentum grid
                for (len_t l = 0; l < Np2P; l++) {
                    const len_t baseP = l * Np1P;
                    for (len_t k = 0; k < Np1P; k++) {
                        const len_t idxP = baseP + k;

                        const real_t S = mollerSMatrix[i * Np1P + k];
                        const real_t delta = EvaluateDeltaMatrixElement(ir, i, k, j, l);
                        sum += Wkl[idxP] * S * delta;
                    }
                }
                sourceVector[idxK] = sum * (scaleFactor / VpK[idxK]);
            }
        }

        offK += mgK->GetNCells();
        offP += mgP->GetNCells();
    }
}

void KnockOnOperatorGeneral::SetMollerSMatrix(real_t *mollerSMatrix) {
    len_t Np_knockon = grid->GetNp1(0);
    len_t Np_primary = grid_primary->GetNp1(0);

    for (len_t i = 0; i < Np_knockon; i++) {
        for (len_t k = 0; k < Np_primary; k++) {
            mollerSMatrix[i * Np_primary + k] =
                KnockOnUtilities::EvaluateMollerFluxMatrixElementOnGrid(
                    i, k, grid, grid_primary, pCutoff
                );
        }
    }
}

void KnockOnOperatorGeneral::
    Rebuild(real_t t, real_t /*dt*/, FVM::UnknownQuantityHandler * /*uqh*/) {
    bool timeHasUpdated = fabs(t - t_source_rebuilt) > 100 * std::numeric_limits<real_t>::epsilon();
    if (timeHasUpdated) {
        // Handle source term explicitly in non-linear time stepper: use at last
        // time step instead of at the current newton iteration
        const real_t *f_primary = unknowns->GetUnknownDataPrevious(id_f_primary);
        SetSourceVector(f_primary);
        t_source_rebuilt = t;
    }
}

// here, x is assumed to be the density to which the source is proportional (ntot)
void KnockOnOperatorGeneral::SetVectorElements(real_t *vec, const real_t *x) {
    len_t offset = 0;
    for (len_t ir = 0; ir < grid->GetNr(); ir++) {
        auto *mg = grid->GetMomentumGrid(ir);
        len_t Np = mg->GetNp1();
        for (len_t i = 0; i < Np; i++)
            for (len_t j = 0; j < mg->GetNp2(); j++) {
                len_t ind = offset + Np * j + i;
                vec[ind] += x[ir] * sourceVector[ind];
            }
        offset += mg->GetNCells();
    }
}

/**
 * Set jacobian matrix elements.
 */
bool KnockOnOperatorGeneral::SetJacobianBlock(
    const len_t /*uqtyId*/, const len_t derivId, FVM::Matrix *jac, const real_t * /*x*/
) {
    if (derivId == id_ntot) {
        len_t offset = 0;
        for (len_t ir = 0; ir < grid->GetNr(); ir++) {
            auto *mg = grid->GetMomentumGrid(ir);
            len_t Np = mg->GetNp1();
            for (len_t i = 0; i < Np; i++)
                for (len_t j = 0; j < mg->GetNp2(); j++) {
                    len_t ind = offset + Np * j + i;
                    jac->SetElement(ind, ir, sourceVector[ind]);
                }
            offset += mg->GetNCells();
        }
        return true;
    }
    return false;
}

bool KnockOnOperatorGeneral::GridRebuilt() {
    Deallocate();
    Allocate();
    return true;
}

// Tabulate the pitch-angle Delta kernel on the reference xi_star grid.
void KnockOnOperatorGeneral::TabulateDeltaMatrixOnXiStarGrid() {
    len_t Nr = grid->GetNr();
    for (len_t ir = 0; ir < Nr; ir++) {
        len_t Nxi = grid->GetNp2(ir);
        for (len_t m = 0; m < n_xi_stars_tabulate; m++) {
            for (len_t l = 0; l < grid_primary->GetNp2(ir); l++) {
                len_t offset = l * Nxi;
                KnockOnUtilities::SetDeltaMatrixColumnOnGrid(
                    ir, xiStarsTab[m], l, grid, grid_primary, deltaTable[ir][m] + offset,
                    n_points_integral
                );
            }
        }
    }
}

/**
 * Interpolates the pre-tabulated orbit-averaged knock-on delta function
 * at a given momentum pair (p, p1) and pitch indices (j, l).
 *
 * Xi_star = EvaluateXiStar(p, p1) is computed and clamped to the
 * tabulated range. Linear interpolation is applied between neighboring
 * xi_star points to preserve the particle conservation encoded in the
 * tabulated delta matrices.
 */
real_t KnockOnOperatorGeneral::EvaluateDeltaMatrixElement(
    len_t ir, len_t i, len_t k, len_t j, len_t l
) {
    const len_t Nxi = grid->GetMomentumGrid(ir)->GetNp2();
    const len_t idxJL = l * Nxi + j;

    auto *mgP = grid_primary->GetMomentumGrid(ir);
    const len_t NpP = mgP->GetNp1();
    const XiStarInterp &T = xiInterp[ir][i * NpP + k];

    if (T.clamp == 1) {
        return deltaTable[ir][0][idxJL];
    }
    if (T.clamp == 2) {
        return deltaTable[ir][n_xi_stars_tabulate - 1][idxJL];
    }

    const len_t m0 = T.m0;
    const real_t w1 = T.w1;
    const real_t w0 = 1.0 - w1;

    const real_t d0 = deltaTable[ir][m0][idxJL];
    const real_t d1 = deltaTable[ir][m0 + 1][idxJL];
    return w0 * d0 + w1 * d1;
}
