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
    len_t id_f_primary, real_t pCutoff, len_t n_xi_stars_tabulate, len_t n_points_integral
)
    : FVM::EquationTerm(grid_knockon),
      grid_primary(grid_primary),
      unknowns(unknowns),
      id_f_primary(id_f_primary),
      pCutoff(pCutoff),
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
    for (len_t ir = 0; ir < Nr; ir++) {
        // XXX: assume p-xi grid
        for (len_t m = 0; m < n_xi_stars_tabulate; m++) {
            delete[] deltaTable[ir][m];
        }
        delete[] deltaTable[ir];
    }
    delete[] deltaTable;
    delete[] xiStarsTab;
    delete[] mollerSMatrix;
    delete[] sourceVector;
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
    SetMollerSMatrix(mollerSMatrix);
    TabulateDeltaMatrixOnXiStarGrid();
}

void KnockOnOperatorGeneral::SetSourceVector(const real_t *f_primary) {
    len_t offset = 0;
    for (len_t ir = 0; ir < grid_primary->GetNr(); ir++) {
        auto *mgK = grid->GetMomentumGrid(ir);
        auto *mgP = grid_primary->GetMomentumGrid(ir);
        for (len_t i = 0; i < mgK->GetNp1(); i++) {
            for (len_t j = 0; j < mgK->GetNp2(); j++) {
                sourceVector[offset + mgK->GetNp1() * j + i] = 0;
                if (grid->GetVpOverP2AtZero(ir)[j]==0){
                    continue;
                }
                for (len_t k = 0; k < mgP->GetNp1(); k++) {
                    for (len_t l = 0; l < mgP->GetNp2(); l++) {
                        real_t Vp1 = grid_primary->GetVp(ir, k, l);
                        real_t f = f_primary[offset + mgP->GetNp1() * l + k];
                        real_t S = mollerSMatrix[i * mgP->GetNp1() + k];
                        real_t delta = EvaluateDeltaMatrixElement(ir, i, k, j, l);
                        real_t dp = mgP->GetDp1(k);
                        real_t dxi = mgP->GetDp2(l);
                        sourceVector[offset + mgK->GetNp1() * j + i] +=
                            dp * dxi * Vp1 * f * S * delta;
                    }
                }
                real_t Vp = grid->GetVp(ir, i, j);
                sourceVector[offset + mgK->GetNp1() * j + i] /= Vp;
            }
        }
        offset += mgK->GetNCells();
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
    const len_t uqtyId, const len_t derivId, FVM::Matrix *jac, const real_t * /*x*/
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
    const FVM::MomentumGrid *mgK = grid->GetMomentumGrid(ir);
    const len_t Nxi = mgK->GetNp2();

    real_t p = mgK->GetP1(i);
    real_t p1 = grid_primary->GetMomentumGrid(ir)->GetP1(k);

    real_t xiStar = KnockOnUtilities::Kinematics::EvaluateXiStar(p, p1);

    // Clamp to tabulation range
    if (xiStar <= xiStar_min) {
        return deltaTable[ir][0][l * Nxi + j];
    }
    if (xiStar >= xiStar_max) {
        return deltaTable[ir][n_xi_stars_tabulate - 1][l * Nxi + j];
    }

    // Map xi_star to fractional index s in [0, n-1]
    real_t s = (xiStar - xiStar_min) / dXiStar;  // since uniform

    // Lower index
    len_t m0 = (len_t)floor(s);

    // Safety clamp (handles rare roundoff making m0==n-1)
    if (m0 >= n_xi_stars_tabulate - 1) m0 = n_xi_stars_tabulate - 2;

    len_t m1 = m0 + 1;

    real_t w1 = s - (real_t)m0;  // in [0,1)
    real_t w0 = 1.0 - w1;

    const real_t d0 = deltaTable[ir][m0][l * Nxi + j];
    const real_t d1 = deltaTable[ir][m1][l * Nxi + j];
    return w0 * d0 + w1 * d1;
}
