/**
 * Unit/integration tests for the KnockOnOperatorGeneral equation term.
 *
 * These tests focus on invariants and structural correctness:
 *  - Interpolated delta-column conservation.
 *  - Linearity in f_primary.
 *  - Consistency between SetVectorElements() and SetJacobianBlock() for ntot.
 *  - Global production identity combining Delta normalization and Moller S conservation.
 *
 * We intentionally allow loose tolerances where appropriate, since production
 * use primarily requires shape correctness and conservation properties.
 */

#include "KnockOnOperatorGeneral.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "DREAM/Equations/Kinetic/KnockOnOperatorGeneral.hpp"
#include "DREAM/Equations/KnockOnUtilities.hpp"
#include "DREAM/Settings/OptionConstants.hpp"
#include "FVM/Grid/Grid.hpp"
#include "FVM/Grid/MomentumGrid.hpp"
#include "FVM/Matrix.hpp"
#include "FVM/UnknownQuantityHandler.hpp"

using namespace DREAMTESTS::_DREAM;

namespace {

constexpr real_t EPS = 5 * std::numeric_limits<real_t>::epsilon();

static void SetPreviousUnknownData(
    DREAM::FVM::UnknownQuantityHandler *uqh, const len_t id, const real_t *data,
    const real_t t = 0.0
) {
    // 1) write to current
    uqh->Store(id, data, /*offs*/ 0, /*mayBeConstant*/ false);

    // 2) promote current->previous
    // 'trueSave' can be false for tests (no need to mark as a true output step)
    uqh->SaveStep(t, /*trueSave*/ false);
}

/**
 * Build a minimal UnknownQuantityHandler containing:
 *  - n_tot (radial scalar)
 *  - E_field (radial scalar) [not used by current operator but often expected present]
 *  - f_primary (distribution on the PRIMARY grid)
 *
 * NOTE: The operator uses GetUnknownDataPrevious(id_f_primary) inside Rebuild(),
 * so the test must ensure the "previous" buffer contains the desired f_primary.
 */
DREAM::FVM::UnknownQuantityHandler *BuildUQH_Minimal(
    DREAM::FVM::Grid *grid_fluid, DREAM::FVM::Grid *grid_primary, len_t &id_ntot, len_t &id_E,
    len_t &id_f_primary
) {
    auto *uqh = new DREAM::FVM::UnknownQuantityHandler();

    // Fluid (radial) unknowns
    id_ntot = uqh->InsertUnknown(DREAM::OptionConstants::UQTY_N_TOT, "0", grid_fluid);
    id_E = uqh->InsertUnknown(DREAM::OptionConstants::UQTY_E_FIELD, "0", grid_fluid);

    // Primary distribution
    id_f_primary = uqh->InsertUnknown(DREAM::OptionConstants::UQTY_F_HOT, "0", grid_primary);

    // Initial values for n_tot and E_field (size = nr)
    real_t *tmp = new real_t[grid_fluid->GetNr()];
    for (len_t ir = 0; ir < grid_fluid->GetNr(); ir++) tmp[ir] = 1.0;
    uqh->SetInitialValue(DREAM::OptionConstants::UQTY_N_TOT, tmp);

    for (len_t ir = 0; ir < grid_fluid->GetNr(); ir++) tmp[ir] = 0.0;
    uqh->SetInitialValue(DREAM::OptionConstants::UQTY_E_FIELD, tmp);
    delete[] tmp;

    // f_primary initial value (size = Ncells(primary))
    {
        const len_t N = grid_primary->GetNCells();
        real_t *f0 = new real_t[N];
        for (len_t i = 0; i < N; i++) f0[i] = 0.0;
        uqh->SetInitialValue(DREAM::OptionConstants::UQTY_F_HOT, f0);
        delete[] f0;
    }

    return uqh;
}

/**
 * Compute total integrated production:
 *   LHS = sum_{i,j} dp dxi Vp(i,j) * sourceVector(i,j)
 *
 * given sourceVector stored on the KNOCKON grid layout.
 */
real_t IntegrateTotalProductionOverKnockonGrid(
    const DREAM::FVM::Grid *grid_knockon, const real_t *sourceVector
) {
    real_t total = 0.0;
    len_t offset = 0;
    for (len_t ir = 0; ir < grid_knockon->GetNr(); ir++) {
        auto *mg = grid_knockon->GetMomentumGrid(ir);
        const len_t Np = mg->GetNp1();
        const len_t Nxi = mg->GetNp2();

        for (len_t i = 0; i < Np; i++) {
            const real_t dp = mg->GetDp1(i);
            for (len_t j = 0; j < Nxi; j++) {
                const real_t dxi = mg->GetDp2(j);
                const len_t ind = offset + Np * j + i;
                const real_t Vp = grid_knockon->GetVp(ir, i, j);
                total += dp * dxi * Vp * sourceVector[ind];
            }
        }

        offset += mg->GetNCells();
    }
    return total;
}

/**
 * Compute RHS prediction using:
 *   RHS = sum_{k,l} dp1 dxi1 Vp1 f(k,l) * [ sum_i dp S_{ik} ]
 *
 * We use KnockOnUtilities::EvaluateMollerFluxMatrixElementOnGrid(i,k,...) for S_{ik}.
 */
real_t PredictTotalProductionFromMollerS(
    const DREAM::FVM::Grid *grid_knockon, const DREAM::FVM::Grid *grid_primary,
    const real_t *f_primary, real_t pCutoff
) {
    real_t total = 0.0;
    len_t offsetP = 0;
    for (len_t ir = 0; ir < grid_primary->GetNr(); ir++) {
        auto *mgK = grid_knockon->GetMomentumGrid(ir);
        auto *mgP = grid_primary->GetMomentumGrid(ir);

        // Precompute sigmaTot(k) = sum_i dp * S_{ik}
        std::vector<real_t> sigmaTot(mgP->GetNp1(), 0.0);
        for (len_t k = 0; k < mgP->GetNp1(); k++) {
            real_t sum = 0.0;
            for (len_t i = 0; i < mgK->GetNp1(); i++) {
                const real_t dp = mgK->GetDp1(i);
                const real_t S = DREAM::KnockOnUtilities::EvaluateMollerFluxMatrixElementOnGrid(
                    i, k, grid_knockon, grid_primary, pCutoff
                );
                sum += dp * S;
            }
            sigmaTot[k] = sum;
        }

        // Now accumulate RHS
        for (len_t k = 0; k < mgP->GetNp1(); k++) {
            const real_t dp1 = mgP->GetDp1(k);
            for (len_t l = 0; l < mgP->GetNp2(); l++) {
                const real_t dxi1 = mgP->GetDp2(l);
                const len_t indP = offsetP + mgP->GetNp1() * l + k;
                const real_t f = f_primary[indP];
                const real_t Vp1 = grid_primary->GetVp(ir, k, l);
                total += dp1 * dxi1 * Vp1 * f * sigmaTot[k];
            }
        }

        offsetP += mgP->GetNCells();
    }

    return total;
}

}  // anonymous namespace

/**
 * Entry point: Run all KnockOnOperatorGeneral tests.
 */
bool KnockOnOperatorGeneral::Run(bool) {
    bool success = true;

    if (CheckKOG_DeltaInterpolationConservation())
        this->PrintOK("KOG: interpolated delta columns preserve normalization.");
    else {
        success = false;
        this->PrintError("KOG test failed: interpolated delta columns do not normalize to 1.");
    }

    if (CheckKOG_SourceVectorLinearity())
        this->PrintOK("KOG: sourceVector is linear in f_primary.");
    else {
        success = false;
        this->PrintError("KOG test failed: sourceVector is not linear in f_primary.");
    }

    if (CheckKOG_JacobianFiniteDifferenceNt())
        this->PrintOK("KOG: Jacobian w.r.t. n_tot matches finite-difference derivative.");
    else {
        success = false;
        this->PrintError("KOG test failed: Jacobian w.r.t. n_tot does not match FD.");
    }

    if (CheckKOG_GlobalProductionIdentity())
        this->PrintOK("KOG: global production identity matches Moller-S prediction.");
    else {
        success = false;
        this->PrintError("KOG test failed: global production identity does not match prediction.");
    }

    return success;
}

/**
 * 1) Interpolated delta column normalization:
 * For random (ir,i,k,l): sum_j dxi_j Delta_interp(j,l; xiStar(p_i,p1_k)) ~= 1
 */
bool KnockOnOperatorGeneral::CheckKOG_DeltaInterpolationConservation() {
    const real_t tol = 5e-2;

    len_t nr = 2;
    len_t npK = 6;
    len_t nxiK = 40;
    len_t npP = 6;
    len_t nxiP = 30;

    len_t ntheta_interp = 50;
    len_t nrProfiles = 8;
    real_t pMin = 0;
    real_t pMax = 3;

    real_t B0 = 1.0;
    auto *gridF = InitializeFluidGrid(nr, B0);
    auto *gridK = InitializeGridGeneralRPXi(nr, npK, nxiK, ntheta_interp, nrProfiles, pMin, pMax);
    auto *gridP = InitializeGridGeneralRPXi(nr, npP, nxiP, ntheta_interp, nrProfiles, pMin, pMax);

    len_t id_ntot, id_E, id_f;
    auto *uqh = BuildUQH_Minimal(gridF, gridP, id_ntot, id_E, id_f);

    const real_t pCutoff = gridK->GetMomentumGrid(0)->GetP1_f(2);  // safe > 0
    DREAM::KnockOnOperatorGeneral op(
        gridK, gridP, uqh, id_f, pCutoff, /*n_xi_stars_tabulate*/ 80, /*n_points_integral*/ 80
    );

    bool success = true;

    for (len_t ir = 0; ir < nr; ir++) {
        auto *mgK = gridK->GetMomentumGrid(ir);
        const len_t NpK = mgK->GetNp1();
        const len_t NxiK = mgK->GetNp2();
        const len_t NxiP = gridP->GetMomentumGrid(ir)->GetNp2();

        // Sample a few indices (keep cheap)
        for (len_t i = 0; i < NpK; i += std::max((len_t)1, NpK / 3)) {
            for (len_t k = 0; k < gridP->GetMomentumGrid(ir)->GetNp1();
                 k += std::max((len_t)1, gridP->GetMomentumGrid(ir)->GetNp1() / 3)) {
                for (len_t l = 0; l < NxiP; l += std::max((len_t)1, NxiP / 4)) {
                    // Skip void cells similarly to your delta tests if needed
                    if (gridP->GetVpOverP2AtZero(ir)[l] == 0) continue;

                    real_t sum = 0.0;
                    for (len_t j = 0; j < NxiK; j++) {
                        const real_t dxi = mgK->GetDp2(j);
                        sum += dxi * op.EvaluateDeltaMatrixElement(ir, i, k, j, l);
                    }

                    if (fabs(sum - 1.0) > tol) {
                        this->PrintError(
                            "Delta normalization failed at ir=%ld i=%ld k=%ld l=%ld: sum=%.8g\n",
                            ir, i, k, l, sum
                        );
                        success = false;
                    }
                }
            }
        }
    }

    delete uqh;
    delete gridF;
    delete gridK;
    delete gridP;

    return success;
}

/**
 * 2) Linearity test:
 * source(fA + fB) == source(fA) + source(fB)
 *
 * This is best done by calling Rebuild() with different "previous" f data and then
 * comparing the operator's internal sourceVector. If sourceVector is private, you
 * can add a lightweight getter in test builds.
 */
bool KnockOnOperatorGeneral::CheckKOG_SourceVectorLinearity() {
    const real_t tol = 1e-10;

    len_t nr = 2;
    len_t np = 6;
    len_t nxi = 20;

    len_t ntheta_interp = 50;
    len_t nrProfiles = 8;
    real_t pMin = 0;
    real_t pMax = 3;

    real_t B0 = 1.0;
    auto *gridF = InitializeFluidGrid(nr, B0);
    auto *gridK = InitializeGridGeneralRPXi(nr, np, nxi, ntheta_interp, nrProfiles, pMin, pMax);
    auto *gridP = InitializeGridGeneralRPXi(nr, np, nxi, ntheta_interp, nrProfiles, pMin, pMax);

    len_t id_ntot, id_E, id_f;
    auto *uqh = BuildUQH_Minimal(gridF, gridP, id_ntot, id_E, id_f);

    const real_t pCutoff = gridK->GetMomentumGrid(0)->GetP1_f(2);
    DREAM::KnockOnOperatorGeneral op(gridK, gridP, uqh, id_f, pCutoff, 60, 80);

    const len_t NpCellsP = gridP->GetNCells();
    real_t *fA = new real_t[NpCellsP];
    real_t *fB = new real_t[NpCellsP];
    real_t *fAB = new real_t[NpCellsP];

    for (len_t i = 0; i < NpCellsP; i++) {
        fA[i] = 0.1 + 0.01 * (real_t)i;
        fB[i] = 0.2 + 0.02 * (real_t)i;
        fAB[i] = fA[i] + fB[i];
    }

    const len_t NK = gridK->GetNCells();
    std::vector<real_t> SA(NK), SB(NK), SAB(NK);

    auto CopySourceK = [&](std::vector<real_t> &dst) {
        const real_t *src = op.GetSourceVector();
        memcpy(dst.data(), src, NK * sizeof(real_t));
    };

    // Run rebuild with fA, capture SA
    SetPreviousUnknownData(uqh, id_f, fA, /*t*/ 0.0);
    op.Rebuild(/*t*/ 0.0, /*dt*/ 1.0, uqh);
    CopySourceK(SA);

    // Run rebuild with fB, capture SB
    SetPreviousUnknownData(uqh, id_f, fB, /*t*/ 1.0);
    op.Rebuild(/*t*/ 1.0, /*dt*/ 1.0, uqh);
    CopySourceK(SB);

    // Run rebuild with fAB, capture SAB
    SetPreviousUnknownData(uqh, id_f, fAB, /*t*/ 2.0);
    op.Rebuild(/*t*/ 2.0, /*dt*/ 1.0, uqh);
    CopySourceK(SAB);
    bool success = true;

    // Compare SAB ~ SA + SB
    len_t offset = 0;
    for (len_t ir=0; ir<gridK->GetNr(); ir++){
        auto *mg = gridK->GetMomentumGrid(ir);
        for (len_t j=0; j<mg->GetNp2(); j++){
            for (len_t i=0; i<mg->GetNp1(); i++){
                len_t idx = offset + mg->GetNp1()*j + i;
                real_t diff = fabs(SAB[idx] - (SA[idx] + SB[idx]));
                if (diff > tol * (1 + fabs(SAB[idx]))) {
                    printf("Test failed (ir=%ld, i=%ld, j=%ld):\n", ir, i, j);
                    printf("  diff = %.8g:\n", diff);
                    printf("  tol = %.8g\n", tol * (1 + fabs(SAB[idx])));
                    printf("  SA = %.8g\n", SA[idx]);
                    printf("  SB = %.8g\n", SB[idx]);
                    printf("  SAB = %.8g\n", SAB[idx]);
                    success = false;
                    break;
                }
            }
        }
        offset += mg->GetNCells();
    }
    delete[] fA;
    delete[] fB;
    delete[] fAB;

    delete uqh;
    delete gridF;
    delete gridK;
    delete gridP;

    return success;
}

/**
 * 3) Jacobian FD test w.r.t. n_tot:
 * Compare SetJacobianBlock(...) against FD derivative of SetVectorElements(...)
 * with respect to n_tot[ir0].
 *
 * This test assumes your term contributes to a kinetic equation row with size Ncells.
 */
bool KnockOnOperatorGeneral::CheckKOG_JacobianFiniteDifferenceNt() {
    const real_t tol = 1e-6;
    const real_t eps = 1e-7;

    len_t nr = 2;
    len_t np = 6;
    len_t nxi = 10;

    len_t ntheta_interp = 50;
    len_t nrProfiles = 8;
    real_t pMin = 0;
    real_t pMax = 3;

    real_t B0 = 1.0;
    auto *gridF = InitializeFluidGrid(nr, B0);
    auto *gridK = InitializeGridGeneralRPXi(nr, np, nxi, ntheta_interp, nrProfiles, pMin, pMax);
    auto *gridP = InitializeGridGeneralRPXi(nr, np, nxi, ntheta_interp, nrProfiles, pMin, pMax);

    len_t id_ntot, id_E, id_f;
    auto *uqh = BuildUQH_Minimal(gridF, gridP, id_ntot, id_E, id_f);

    const real_t pCutoff = gridK->GetMomentumGrid(0)->GetP1_f(2);
    DREAM::KnockOnOperatorGeneral op(gridK, gridP, uqh, id_f, pCutoff, 60, 80);

    // Set some f_primary in previous and rebuild source
    const len_t NP = gridP->GetNCells();
    real_t *f0 = new real_t[NP];
    for (len_t i = 0; i < NP; i++) f0[i] = 1e-3;
    SetPreviousUnknownData(uqh, id_f, f0, /*t*/ 0.0);
    op.Rebuild(/*t*/ 0.0, /*dt*/ 1.0, uqh);

    // Residual vectors
    const len_t NK = gridK->GetNCells();
    real_t *R0 = new real_t[NK];
    real_t *R1 = new real_t[NK];
    for (len_t q = 0; q < NK; q++) {
        R0[q] = 0.0;
        R1[q] = 0.0;
    }

    // Grab ntot pointer and perturb one radius. If uqh doesn't expose direct writing,
    // replace with setter calls.
    real_t *nt = new real_t[nr];
    for (len_t ir = 0; ir < nr; ir++) nt[ir] = 1.0;

    // Evaluate residual at nt
    op.SetVectorElements(R0, /*x*/ nt);

    // Perturb nt at ir0
    len_t ir0 = 0;
    real_t old = nt[ir0];
    nt[ir0] = old + eps;

    op.SetVectorElements(R1, /*x*/ nt);

    // FD derivative vector dR/dnt_ir0
    std::vector<real_t> dR(NK, 0.0);
    for (len_t q = 0; q < NK; q++) dR[q] = (R1[q] - R0[q]) / eps;

    // Build Jacobian column using SetJacobianBlock
    // Jacobian is size NK x nr for this coupling
    DREAM::FVM::Matrix jac(NK, NK, 1);
    jac.Zero();
    jac.PartialAssemble();
    op.SetJacobianBlock(id_f, id_ntot, &jac, /*x*/ nullptr);
    jac.Assemble();
    // Compare column ir0
    bool success = true;
    for (len_t q = 0; q < NK; q++) {
        real_t J = jac.GetElement(q, ir0);
        real_t diff = fabs(J - dR[q]);
        if (diff > tol * (1 + fabs(J) + fabs(dR[q]))) {
            success = false;
            break;
        }
    }

    delete[] f0;
    delete[] R0;
    delete[] R1;
    delete[] nt;

    delete uqh;
    delete gridF;
    delete gridK;
    delete gridP;

    return success;
}

/**
 * 4) Global production identity:
 * LHS = ∑_{i,j} dp dxi Vp * sourceVector(i,j)
 * RHS = ∑_{k,l} dp1 dxi1 Vp1 f(k,l) * [∑_i dp S_{ik}]
 *
 * This combines Delta normalization and Moller S conservation into a single
 * integration test for the assembled source.
 */
bool KnockOnOperatorGeneral::CheckKOG_GlobalProductionIdentity() {
    const real_t rtol = 5e-2;
    const real_t atol = 1e-10;

    len_t nr = 2;
    len_t np = 20;
    len_t nxi = 10;

    len_t ntheta_interp = 50;
    len_t nrProfiles = 8;
    real_t pMin = 0;
    real_t pMax = 4;

    real_t B0 = 1.0;
    auto *gridF = InitializeFluidGrid(nr, B0);
    auto *gridK = InitializeGridGeneralRPXi(nr, np, nxi, ntheta_interp, nrProfiles, pMin, pMax);
    auto *gridP = InitializeGridGeneralRPXi(nr, np, nxi, ntheta_interp, nrProfiles, pMin, pMax);

    len_t id_ntot, id_E, id_f;
    auto *uqh = BuildUQH_Minimal(gridF, gridP, id_ntot, id_E, id_f);

    const real_t pCutoff = gridK->GetMomentumGrid(0)->GetP1_f(2);
    DREAM::KnockOnOperatorGeneral op(gridK, gridP, uqh, id_f, pCutoff, 80, 80);

    // Choose a simple positive f_primary
    const len_t NP = gridP->GetNCells();
    real_t *f = new real_t[NP];
    for (len_t i = 0; i < NP; i++) {
        f[i] = 1e-3;
    }

    // Install as "previous" and rebuild
    SetPreviousUnknownData(uqh, id_f, f, /*t*/ 0.0);
    op.Rebuild(/*t*/ 0.0, /*dt*/ 1.0, uqh);

    // Get operator internal sourceVector. Add a test-only getter if needed.
    // const real_t *source = op.GetSourceVector();
    const real_t *source = op.GetSourceVector();

    // LHS: integrate over knock-on phase space with Vp measure
    real_t LHS = 0.0;
    if (source != nullptr) LHS = IntegrateTotalProductionOverKnockonGrid(gridK, source);

    // RHS: predicted from sigmaTot(k) and integrated primary weight
    real_t RHS = PredictTotalProductionFromMollerS(gridK, gridP, f, pCutoff);

    bool success = true;
    if (source == nullptr) {
        // Until you expose sourceVector, the test can't run fully.
        // Treat as failure so you remember to wire it.
        success = false;
    } else {
        const real_t diff = fabs(LHS - RHS);
        if (diff > atol + rtol * (fabs(LHS) + fabs(RHS))) {
            this->PrintError("Global identity failed:\n");
            this->PrintError("  LHS: %.8g\n", LHS);
            this->PrintError("  RHS: %.8g\n", RHS);
            this->PrintError("  diff: %.8g\n", diff);
            success = false;
        }
    }

    delete[] f;

    delete uqh;
    delete gridF;
    delete gridK;
    delete gridP;

    return success;
}
