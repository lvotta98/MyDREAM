/**
 * Unit/integration tests for the KnockOnOperatorGeneral equation term.
 *
 * The tests are written against the public EquationTerm interface:
 *  - Rebuild() (explicit-time cache behavior)
 *  - SetVectorElements() (assembled contribution)
 *  - SetJacobianBlock() (linearization w.r.t. n_tot)
 *
 * We intentionally allow modest tolerances in some integration tests since the
 * production requirement is conservation/shape correctness rather than tight pointwise accuracy.
 */

#include "KnockOnOperatorGeneral.hpp"

#include <cmath>
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

/**
 * Store data into the unknown handler such that GetUnknownDataPrevious(id) returns it.
 *
 * NOTE: SaveStep() may be time-sensitive; tests that rely on updating "previous" should
 * pass a strictly increasing t to ensure the previous buffer is actually advanced.
 */
void SetPreviousUnknownData(
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
        for (len_t idx = 0; idx < N; idx++) {
            f0[idx] = 0.0;
        }
        uqh->SetInitialValue(DREAM::OptionConstants::UQTY_F_HOT, f0);
        delete[] f0;
    }

    return uqh;
}

/**
 * Minimal UnknownQuantityHandler for cross-grid knock-on tests with runaway primaries.
 *
 * Contains:
 *  - n_tot  (fluid grid, radial scalar)
 *  - E_field (fluid grid, radial scalar; unused but commonly present)
 *  - f_re   (runaway grid; serves as f_primary for KnockOnOperatorGeneral)
 */
DREAM::FVM::UnknownQuantityHandler *BuildUQH_MinimalFRe(
    DREAM::FVM::Grid *grid_fluid, DREAM::FVM::Grid *grid_runaway, len_t &id_ntot, len_t &id_E,
    len_t &id_f_re
) {
    auto *uqh = new DREAM::FVM::UnknownQuantityHandler();

    id_ntot = uqh->InsertUnknown(DREAM::OptionConstants::UQTY_N_TOT, "0", grid_fluid);
    id_E = uqh->InsertUnknown(DREAM::OptionConstants::UQTY_E_FIELD, "0", grid_fluid);

    // Runaway distribution (primaries)
    id_f_re = uqh->InsertUnknown(DREAM::OptionConstants::UQTY_F_RE, "0", grid_runaway);

    // n_tot initial value
    {
        real_t *tmp = new real_t[grid_fluid->GetNr()];
        for (len_t ir = 0; ir < grid_fluid->GetNr(); ir++) tmp[ir] = 1.0;
        uqh->SetInitialValue(DREAM::OptionConstants::UQTY_N_TOT, tmp);

        for (len_t ir = 0; ir < grid_fluid->GetNr(); ir++) tmp[ir] = 0.0;
        uqh->SetInitialValue(DREAM::OptionConstants::UQTY_E_FIELD, tmp);
        delete[] tmp;
    }

    // f_re initial value
    {
        const len_t N = grid_runaway->GetNCells();
        real_t *f0 = new real_t[N];
        for (len_t idx = 0; idx < N; idx++) {
            f0[idx] = 0.0;
        }
        uqh->SetInitialValue(DREAM::OptionConstants::UQTY_F_RE, f0);
        delete[] f0;
    }

    return uqh;
}

/**
 * Integrate an assembled knock-on contribution over (p,xi) phase space:
 *   ∑_{i,j} dp dxi Vp(i,j) * vec(i,j)
 *
 * The input vector must be stored on the knock-on grid layout (Np-major with pitch blocks),
 * i.e. ind = offset + Np*j + i.
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
 * We use KnockOnUtilities::EvaluateMollerFluxIntegratedOverKnockonGrid(k)
 * for the total cross section.
 */
real_t PredictTotalProductionFromMollerS(
    const DREAM::FVM::Grid *grid_knockon, const DREAM::FVM::Grid *grid_primary,
    const real_t *f_primary, real_t pCutoff
) {
    real_t total = 0.0;
    len_t offsetP = 0;
    for (len_t ir = 0; ir < grid_primary->GetNr(); ir++) {
        auto *mgP = grid_primary->GetMomentumGrid(ir);
        // Now accumulate RHS
        for (len_t k = 0; k < mgP->GetNp1(); k++) {
            const real_t dp1 = mgP->GetDp1(k);
            real_t sigmaTot = DREAM::KnockOnUtilities::EvaluateMollerFluxIntegratedOverKnockonGrid(
                k, grid_knockon, grid_primary, pCutoff
            );
            for (len_t l = 0; l < mgP->GetNp2(); l++) {
                const real_t dxi1 = mgP->GetDp2(l);
                const len_t indP = offsetP + mgP->GetNp1() * l + k;
                const real_t f = f_primary[indP];
                const real_t Vp1 = grid_primary->GetVp(ir, k, l);
                total += dp1 * dxi1 * Vp1 * f * sigmaTot;
            }
        }

        offsetP += mgP->GetNCells();
    }

    return total;
}

/**
 * Apply this term's contribution for a spatially constant n_tot profile.
 * This calls SetVectorElements() into a zeroed vector so the result contains only this term.
 */
void ApplyTerm(
    DREAM::KnockOnOperatorGeneral &op, const DREAM::FVM::Grid *grid_knockon, const real_t ntot,
    std::vector<real_t> &out
) {
    len_t Nr = grid_knockon->GetNr();
    std::vector<real_t> ntot_arr;
    ntot_arr.assign(Nr, ntot);
    const len_t NK = grid_knockon->GetNCells();
    out.assign(NK, 0.0);  // ensure we test *this term only*
    op.SetVectorElements(out.data(), ntot_arr.data());
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

    if (CheckKOG_MollerKernelConservation())
        this->PrintOK("KOG: Moller kernel integrates to total cross section.");
    else {
        success = false;
        this->PrintError("KOG test failed: Moller kernel does not integrate to total cross section."
        );
    }

    if (CheckKOG_LinearityInFPrimary())
        this->PrintOK("KOG: assembled contribution is linear in f_primary.");
    else {
        success = false;
        this->PrintError("KOG test failed: assembled contribution is not linear in f_primary.");
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
    if (CheckKOG_HotRunawayGlobalProductionIdentity())
        this->PrintOK("KOG: hot<-runaway (cross-grid) global identity passed.");
    else {
        success = false;
        this->PrintError("KOG test failed: hot<-runaway (cross-grid) global identity.");
    }
    if (CheckKOG_TimeCachingRegression())
        this->PrintOK("KOG: time caching regression test passed.");
    else {
        success = false;
        this->PrintError("KOG test failed: time caching regression.");
    }
    if (CheckKOG_NonNegativity())
        this->PrintOK("KOG: non-negativity sanity check passed.");
    else {
        success = false;
        this->PrintError("KOG test failed: non-negativity sanity check.");
    }
    if (CheckKOG_RadiusLocality())
        this->PrintOK("KOG: radius locality test passed.");
    else {
        success = false;
        this->PrintError("KOG test failed: radius locality.");
    }

    return success;
}

/**
 * Delta column normalization using the same accumulation path as assembly:
 *
 * For sampled (ir,i,k,l), set W_l = e_l and Sik=1 so that
 *   outPitch_j[j] = Δ_{j l}(xi*(p_i,p1_k))
 * and verify:
 *   Σ_j dxi_j outPitch_j[j] = 1   (for non-void primary pitch cells l)
 *   Σ_j dxi_j outPitch_j[j] = 0   (for void primary pitch cells l)
 *
 * Effectively extracts the angle kernel by setting sigma=1 and the distribution
 * as a delta function at each pitch.
 */
bool KnockOnOperatorGeneral::CheckKOG_DeltaInterpolationConservation() {
    const real_t tol = 1e-14;

    len_t nr = 2;
    len_t npK = 5;
    len_t nxiK = 40;
    len_t npP = 7;
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

    const real_t pCutoff = gridK->GetMomentumGrid(0)->GetP1(1);
    constexpr real_t scalefactor = 1.0;
    constexpr len_t n_xi_stars_tabulate = 80;
    constexpr len_t n_points_integral = 80;
    DREAM::KnockOnOperatorGeneral op(
        gridK, gridP, uqh, id_f, pCutoff, scalefactor, n_xi_stars_tabulate, n_points_integral
    );

    bool success = true;

    for (len_t ir = 0; ir < nr; ir++) {
        auto *mgK = gridK->GetMomentumGrid(ir);
        auto *mgP = gridP->GetMomentumGrid(ir);

        const len_t NpK = mgK->GetNp1();
        const len_t NxiK = mgK->GetNp2();
        const len_t NpP = mgP->GetNp1();
        const len_t NxiP = mgP->GetNp2();

        // Scratch arrays
        std::vector<real_t> W_l(NxiP, 0.0);
        std::vector<real_t> outPitch(NxiK, 0.0);

        for (len_t i = 0; i < NpK; i++) {
            for (len_t k = 0; k < NpP; k++) {
                for (len_t l = 0; l < NxiP; l++) {
                    // W_l = e_l to select exactly column l
                    std::fill(W_l.begin(), W_l.end(), 0.0);
                    W_l[l] = 1.0;

                    // outPitch = 0 then accumulate using the same interpolation path as assembly
                    std::fill(outPitch.begin(), outPitch.end(), 0.0);
                    op.AccumulateAngleKernel(ir, i, k, W_l.data(), /*Sik=*/1.0, outPitch.data());

                    // Integral over knock-on pitch: Σ_j dxi_j * Δ_{j l}
                    real_t sum = 0.0;
                    for (len_t j = 0; j < NxiK; j++) {
                        sum += mgK->GetDp2(j) * outPitch[j];
                    }

                    const bool primaryVoid = (gridP->GetVpOverP2AtZero(ir)[l] == 0);
                    if (primaryVoid) {
                        // should contribute nothing at all
                        if (fabs(sum) != 0) {
                            this->PrintError(
                                "Delta normalization non-zero in void primary cell at ir=%ld i=%ld "
                                "k=%ld l=%ld: sum=%.16g",
                                ir, i, k, l, sum
                            );
                            success = false;
                        }
                    } else {
                        // should integrate to 1 exactly (up to tol)
                        if (fabs(sum - 1.0) > tol) {
                            this->PrintError(
                                "Delta normalization failed at ir=%ld i=%ld k=%ld l=%ld: sum=%.16g",
                                ir, i, k, l, sum
                            );
                            success = false;
                        }
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

bool KnockOnOperatorGeneral::CheckKOG_MollerKernelConservation() {
    const real_t tol = 1e-14;

    len_t nr = 2;
    len_t npK = 40;
    len_t nxiK = 9;
    len_t npP = 25;
    len_t nxiP = 11;

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
    constexpr real_t scalefactor = 1.0;
    constexpr len_t n_xi_stars_tabulate = 80;
    constexpr len_t n_points_integral = 80;
    DREAM::KnockOnOperatorGeneral op(
        gridK, gridP, uqh, id_f, pCutoff, scalefactor, n_xi_stars_tabulate, n_points_integral
    );

    bool success = true;

    for (len_t ir = 0; ir < nr; ir++) {
        auto *mgK = gridK->GetMomentumGrid(ir);
        auto *mgP = gridP->GetMomentumGrid(ir);
        const len_t NpK = mgK->GetNp1();
        const len_t NpP = mgP->GetNp1();
        const real_t *dpK = mgK->GetDp1();
        // Sample a few indices (keep cheap)
        for (len_t k = 0; k < NpP; k++) {
            real_t sum = 0.0;
            for (len_t i = 0; i < NpK; i++) {
                sum += dpK[i] * op.MollerDifferentialCrossSection(ir, i, k);
            }
            real_t expected = DREAM::KnockOnUtilities::EvaluateMollerFluxIntegratedOverKnockonGrid(
                k, gridK, gridP, pCutoff
            );
            if (fabs(sum - expected) > tol) {
                this->PrintError("Failed at k=%ld", k);
                this->PrintError("sum: %.16g", sum);
                this->PrintError("expected: %.16g", expected);
                success = false;
            }
        }
    }
    return success;
}

/**
 * Linearity in f_primary:
 * For fixed n_tot, the assembled contribution satisfies:
 *   F(fA + fB) ≈ F(fA) + F(fB),
 * where F is obtained via Rebuild()+SetVectorElements().
 */
bool KnockOnOperatorGeneral::CheckKOG_LinearityInFPrimary() {
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

    const real_t scaleFactor = 1.0;
    const len_t nXiStars = 60;
    const len_t nIntPts = 80;
    DREAM::KnockOnOperatorGeneral op(
        gridK, gridP, uqh, id_f, pCutoff, scaleFactor, nXiStars, nIntPts
    );

    // Build fA, fB, fA+fB
    const len_t NP = gridP->GetNCells();
    real_t *fA = new real_t[NP];
    real_t *fB = new real_t[NP];
    real_t *fAB = new real_t[NP];
    for (len_t i = 0; i < NP; i++) {
        fA[i] = 1e20 + 1e18 * (real_t)i;
        fB[i] = 1e20 + 2e18 * (real_t)i;
        fAB[i] = fA[i] + fB[i];
    }

    real_t ntot = 1;
    const len_t NK = gridK->GetNCells();
    std::vector<real_t> RA, RB, RAB;

    // fA
    SetPreviousUnknownData(uqh, id_f, fA, /*t*/ 0.0);
    op.Rebuild(/*t*/ 0.0, /*dt*/ 1.0, uqh);
    ApplyTerm(op, gridK, ntot, RA);

    // fB
    SetPreviousUnknownData(uqh, id_f, fB, /*t*/ 1.0);
    op.Rebuild(/*t*/ 1.0, /*dt*/ 1.0, uqh);
    ApplyTerm(op, gridK, ntot, RB);

    // fA+fB
    SetPreviousUnknownData(uqh, id_f, fAB, /*t*/ 2.0);
    op.Rebuild(/*t*/ 2.0, /*dt*/ 1.0, uqh);
    ApplyTerm(op, gridK, ntot, RAB);

    bool success = true;
    for (len_t q = 0; q < NK; q++) {
        const real_t lhs = RAB[q];
        const real_t rhs = RA[q] + RB[q];
        const real_t diff = fabs(lhs - rhs);
        if (diff > tol * (1 + fabs(lhs))) {
            this->PrintError(
                "Linearity failed at q=%ld: lhs=%.8g rhs=%.8g diff=%.8g", (long)q, lhs, rhs, diff
            );
            success = false;
            break;
        }
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
    constexpr real_t scalefactor = 1.0;
    constexpr len_t n_xi_stars_tabulate = 80;
    constexpr len_t n_points_integral = 80;
    DREAM::KnockOnOperatorGeneral op(
        gridK, gridP, uqh, id_f, pCutoff, scalefactor, n_xi_stars_tabulate, n_points_integral
    );

    // Set some f_primary in previous and rebuild source
    const len_t NP = gridP->GetNCells();
    real_t *f0 = new real_t[NP];
    for (len_t i = 0; i < NP; i++) f0[i] = 1e20;
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
 * Global production identity (integration test):
 * LHS = ∑_{i,j} dp dxi Vp * vec(i,j),  with vec from SetVectorElements() at n_tot=1.
 * RHS = ∑_{k,l} dp1 dxi1 Vp1 f(k,l) * [∑_i dp S_{ik}]
 *
 * This combines delta normalization and Møller S "flux" conservation into a single check
 * on the assembled source term.
 */
bool KnockOnOperatorGeneral::CheckKOG_GlobalProductionIdentity() {
    const real_t rtol = 1e-14;
    const real_t atol = 1e-14;

    len_t nr = 2;
    len_t np = 15;
    len_t nxi = 30;

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

    auto *mgK = gridK->GetMomentumGrid(0);
    const real_t pCutoff = mgK->GetP1_f(2);

    const real_t scaleFactor = 1.0;
    const len_t nXiStars = 80;
    const len_t nIntPts = 80;
    DREAM::KnockOnOperatorGeneral op(
        gridK, gridP, uqh, id_f, pCutoff, scaleFactor, nXiStars, nIntPts
    );

    // f_primary >= 0
    const len_t NP = gridP->GetNCells();
    real_t *f = new real_t[NP];
    for (len_t i = 0; i < NP; i++) {
        f[i] = 1e20;
    }
    // Install as previous and rebuild cached source
    SetPreviousUnknownData(uqh, id_f, f, /*t*/ 0.0);
    op.Rebuild(/*t*/ 0.0, /*dt*/ 1.0, uqh);

    real_t ntot = 1;
    std::vector<real_t> vec;
    ApplyTerm(op, gridK, ntot, vec);

    // LHS: integrate vec over knock-on phase space with Vp measure
    const real_t LHS = IntegrateTotalProductionOverKnockonGrid(gridK, vec.data());

    // RHS: predicted production from Moller S (same f_primary) times scaleFactor
    const real_t RHS0 = PredictTotalProductionFromMollerS(gridK, gridP, f, pCutoff);
    const real_t RHS = scaleFactor * RHS0;
    const real_t diff = fabs(LHS - RHS);
    const bool success = (diff <= atol + rtol * (fabs(LHS) + fabs(RHS)));
    if (!success) {
        this->PrintError("Global identity failed:");
        this->PrintError("  LHS: %.8g", LHS);
        this->PrintError("  RHS: %.8g", RHS);
        this->PrintError("  diff: %.8g", diff);
    }

    delete[] f;

    delete uqh;
    delete gridF;
    delete gridK;
    delete gridP;

    return success;
}

bool KnockOnOperatorGeneral::CheckKOG_HotRunawayGlobalProductionIdentity() {
    const real_t rtol = 1e-14;
    const real_t atol = 1e-14;
    const real_t absTolNonNeg = 1e-13;

    // Keep grids modest for runtime but not tiny.
    const len_t nr = 2;

    const len_t npHot = 18;
    const len_t nxiHot = 12;

    const len_t npRe = 22;
    const len_t nxiRe = 14;

    const len_t ntheta_interp = 50;
    const len_t nrProfiles = 8;

    const real_t pMin = 0.0;
    const real_t pMaxHot = 4.0;
    const real_t pMaxRe = 20.0;

    const real_t B0 = 1.0;
    auto *gridF = InitializeFluidGrid(nr, B0);

    // Knock-on grid: hottail
    auto *gridHot =
        InitializeGridGeneralRPXi(nr, npHot, nxiHot, ntheta_interp, nrProfiles, pMin, pMaxHot);

    // Primary grid: runaway (extends to higher p)
    auto *gridRe =
        InitializeGridGeneralRPXi(nr, npRe, nxiRe, ntheta_interp, nrProfiles, pMin, pMaxRe);

    len_t id_ntot, id_E, id_f_re;
    auto *uqh = BuildUQH_MinimalFRe(gridF, gridRe, id_ntot, id_E, id_f_re);

    // Choose a cutoff safely >0 and within the hot grid range.
    const real_t pCutoff = gridHot->GetMomentumGrid(0)->GetP1_f(2);

    const real_t ntot = 1.0;
    const real_t scaleFactor = 1.0;
    const len_t nXiStars = 80;
    const len_t nIntPts = 80;

    // This is exactly the cross term you assemble in f_hot.cpp:
    // source onto hottail grid from primaries living on runaway grid.
    DREAM::KnockOnOperatorGeneral op(
        gridHot, gridRe, uqh, id_f_re, pCutoff, scaleFactor, nXiStars, nIntPts
    );

    // Build a positive runaway primary distribution.
    const len_t NRe = gridRe->GetNCells();
    real_t *f = new real_t[NRe];
    for (len_t idx = 0; idx < NRe; idx++) {
        // mildly varying to avoid accidental symmetry
        f[idx] = 1e20 * (1.0 + 0.1 * (real_t)(idx % 7));
    }

    // Ensure "previous" really updates (SaveStep can be time-sensitive).
    const real_t dt = 1.0;
    const real_t t = 1.0;
    SetPreviousUnknownData(uqh, id_f_re, f, t - dt);

    op.Rebuild(t, dt, uqh);

    // Apply with n_tot = 1 everywhere
    std::vector<real_t> vec;
    ApplyTerm(op, gridHot, ntot, vec);

    bool success = true;

    // Quick non-negativity sanity (should hold up to tiny numerical noise)
    {
        real_t minVal = 1e300;
        len_t minIdx = 0;
        for (len_t q = 0; q < (len_t)vec.size(); q++) {
            if (vec[q] < minVal) {
                minVal = vec[q];
                minIdx = q;
            }
        }
        if (minVal < -absTolNonNeg) {
            this->PrintError(
                "hot<-re non-negativity failed: min(vec)=%.8g at q=%ld", minVal, minIdx
            );
            success = false;
        }
    }

    // Global identity:
    // LHS = ∫ vec over knock-on (hot) phase space
    const real_t LHS = IntegrateTotalProductionOverKnockonGrid(gridHot, vec.data());

    // RHS = ∫ f_re over primary phase space, weighted by sigmaTot(k)=∫ S_{ik} dp
    const real_t RHS0 = PredictTotalProductionFromMollerS(gridHot, gridRe, f, pCutoff);
    const real_t RHS = scaleFactor * ntot * RHS0;
    const real_t diff = fabs(LHS - RHS);
    if (diff > atol + rtol * (fabs(LHS) + fabs(RHS))) {
        this->PrintError("hot<-re global identity failed:");
        this->PrintError("  LHS: %.8g", LHS);
        this->PrintError("  RHS: %.8g", RHS);
        this->PrintError("  diff: %.8g", diff);
        success = false;
    }

    delete[] f;

    delete uqh;
    delete gridF;
    delete gridHot;
    delete gridRe;

    return success;
}

/**
 * Explicit-time caching regression:
 * Rebuild() should only refresh the cached source when t changes.
 * This test also verifies linear scaling of the assembled contribution when f_primary is scaled.
 */
bool KnockOnOperatorGeneral::CheckKOG_TimeCachingRegression() {
    const real_t rtol = 1e-12;

    len_t nr = 2;
    len_t np = 6;
    len_t nxi = 12;

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
    const real_t scaleFactor = 1.0;
    const len_t nXiStars = 60;
    const len_t nIntPts = 60;
    DREAM::KnockOnOperatorGeneral op(
        gridK, gridP, uqh, id_f, pCutoff, scaleFactor, nXiStars, nIntPts
    );

    const len_t NP = gridP->GetNCells();
    real_t *fA = new real_t[NP];
    real_t *fB = new real_t[NP];

    // Make sure the two cases differ clearly
    for (len_t idx = 0; idx < NP; idx++) {
        fA[idx] = 1e20;
        fB[idx] = 2e20;
    }

    std::vector<real_t> vecA, vecSameT, vecNewT;
    real_t ntot = 1.0;

    real_t dt = 1.0;
    real_t t1 = 1.0;
    real_t t2 = 2.0;
    // Build with fA at t=0
    SetPreviousUnknownData(uqh, id_f, fA, t1 - dt);
    op.Rebuild(t1, dt, uqh);
    ApplyTerm(op, gridK, ntot, vecA);

    // Update previous buffer to fB but keep the same t=0 -> should NOT rebuild
    SetPreviousUnknownData(uqh, id_f, fB, t1);
    op.Rebuild(t1, dt, uqh);
    ApplyTerm(op, gridK, ntot, vecSameT);

    // Now change time -> should rebuild using fB
    op.Rebuild(t2, dt, uqh);
    ApplyTerm(op, gridK, ntot, vecNewT);

    // Check vecSameT == vecA (cache held)
    real_t maxRef = 0.0, maxDiffSame = 0.0, maxDiffNew = 0.0;
    for (len_t q = 0; q < (len_t)vecA.size(); q++) {
        maxRef = std::max(maxRef, fabs(vecA[q]));
        maxDiffSame = std::max(maxDiffSame, fabs(vecSameT[q] - vecA[q]));
        maxDiffNew = std::max(maxDiffNew, fabs(vecNewT[q] - 2.0 * vecA[q]));  // since fB=2*fA
    }

    bool success = true;
    if (maxDiffSame > rtol * (1.0 + maxRef)) {
        this->PrintError("Time-caching failed: vec changed despite same t.");
        this->PrintError("  maxDiffSame=%.8g maxRef=%.8g", maxDiffSame, maxRef);
        success = false;
    }
    if (maxDiffNew > rtol * (1.0 + 2.0 * maxRef)) {
        this->PrintError("Time-caching failed: vec after t-change not consistent with f scaling.");
        this->PrintError("  maxDiffNew=%.8g maxRef=%.8g", maxDiffNew, maxRef);
        success = false;
    }

    delete[] fA;
    delete[] fB;

    delete uqh;
    delete gridF;
    delete gridK;
    delete gridP;

    return success;
}

/**
 * Non-negativity sanity:
 * For f_primary >= 0 and n_tot >= 0, the assembled contribution should be >= 0
 * up to small numerical noise from interpolation/quadrature.
 */
bool KnockOnOperatorGeneral::CheckKOG_NonNegativity() {
    // allow tiny negative due to interpolation/roundoff
    const real_t absTol = 1e-13;

    len_t nr = 2;
    len_t np = 10;
    len_t nxi = 16;

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
    const real_t scaleFactor = 1.0;
    const len_t nXiStars = 60;
    const len_t nIntPts = 60;
    DREAM::KnockOnOperatorGeneral op(
        gridK, gridP, uqh, id_f, pCutoff, scaleFactor, nXiStars, nIntPts
    );

    // Positive f_primary
    const len_t NP = gridP->GetNCells();
    real_t *f = new real_t[NP];
    for (len_t idx = 0; idx < NP; idx++) {
        f[idx] = 1e20;
    }
    SetPreviousUnknownData(uqh, id_f, f, /*t*/ 0.0);
    op.Rebuild(/*t*/ 0.0, /*dt*/ 1.0, uqh);

    std::vector<real_t> vec;
    ApplyTerm(op, gridK, /*ntot*/ 1.0, vec);

    bool success = true;
    real_t minVal = 1e300;
    len_t minIdx = 0;

    for (len_t q = 0; q < (len_t)vec.size(); q++) {
        if (vec[q] < minVal) {
            minVal = vec[q];
            minIdx = q;
        }
    }

    if (minVal < -absTol) {
        this->PrintError("Non-negativity failed: min(vec)=%.8g at q=%ld", minVal, (long)minIdx);
        success = false;
    }

    // check source==0 in void region (Vp==0)
    len_t offset = 0;
    for (len_t ir = 0; ir < gridK->GetNr(); ir++) {
        for (len_t i = 0; i < gridK->GetNp1(ir); i++) {
            for (len_t j = 0; j < gridK->GetNp2(ir); j++) {
                len_t idx = offset + gridK->GetNp1(ir) * j + i;
                if (gridK->GetVp(ir, i, j) == 0 && vec[idx] != 0) {
                    this->PrintError(
                        "Non-negativity failed: source is non-zero in void region (Vp==0): "
                        "vec=%.16g at ir=%ld, i=%ld, j=%ld",
                        vec[i], ir, i, j
                    );
                    success = false;
                }
            }
        }
        offset += gridK->GetMomentumGrid(ir)->GetNCells();
    }

    delete[] f;

    delete uqh;
    delete gridF;
    delete gridK;
    delete gridP;

    return success;
}

/**
 * Radius locality:
 * Only the phase-space block belonging to radius ir depends on n_tot[ir].
 * Setting n_tot nonzero at a single radius must not affect other radius blocks.
 */
bool KnockOnOperatorGeneral::CheckKOG_RadiusLocality() {
    const real_t absTol = 1e-14;
    const real_t relTol = 1e-12;

    len_t nr = 3;
    len_t np = 8;
    len_t nxi = 12;

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
    const real_t scaleFactor = 1.0;
    const len_t nXiStars = 60;
    const len_t nIntPts = 60;
    DREAM::KnockOnOperatorGeneral op(
        gridK, gridP, uqh, id_f, pCutoff, scaleFactor, nXiStars, nIntPts
    );

    // Positive f_primary
    const len_t NP = gridP->GetNCells();
    real_t *f = new real_t[NP];
    for (len_t idx = 0; idx < NP; idx++) {
        f[idx] = 1e20;
    }

    SetPreviousUnknownData(uqh, id_f, f, /*t*/ 0.0);
    op.Rebuild(/*t*/ 0.0, /*dt*/ 1.0, uqh);

    // vecAll: ntot=1 everywhere (use your helper)
    std::vector<real_t> vecAll;
    ApplyTerm(op, gridK, /*ntot*/ 1.0, vecAll);

    // vecLocal: ntot nonzero only at ir0
    const len_t NK = gridK->GetNCells();
    std::vector<real_t> vecLocal(NK, 0.0);
    std::vector<real_t> ntot_arr(nr, 0.0);

    const len_t ir0 = 1;
    ntot_arr[ir0] = 1.0;
    op.SetVectorElements(vecLocal.data(), ntot_arr.data());

    // Check locality by radius blocks
    bool success = true;
    len_t offset = 0;
    for (len_t ir = 0; ir < nr; ir++) {
        const auto *mg = gridK->GetMomentumGrid(ir);
        const len_t nCells = mg->GetNCells();
        const len_t start = offset;
        const len_t end = offset + nCells;

        if (ir != ir0) {
            // Outside ir0 block: must be ~0
            real_t maxAbs = 0.0;
            for (len_t q = start; q < end; q++) {
                const real_t a = fabs(vecLocal[q]);
                if (a > maxAbs) maxAbs = a;
            }
            if (maxAbs > absTol) {
                this->PrintError(
                    "Radius locality failed: ir=%ld has leakage maxAbs=%.8g", (long)ir, maxAbs
                );
                success = false;
                break;
            }
        } else {
            // Inside ir0 block: should match vecAll (since ntot=1 there too)
            real_t maxDiff = 0.0;
            real_t maxRef = 0.0;
            for (len_t q = start; q < end; q++) {
                const real_t diff = fabs(vecLocal[q] - vecAll[q]);
                if (diff > maxDiff) maxDiff = diff;
                const real_t ref = fabs(vecAll[q]);
                if (ref > maxRef) maxRef = ref;
            }
            const real_t thresh = absTol + relTol * (1.0 + maxRef);
            if (maxDiff > thresh) {
                this->PrintError("Radius locality failed: ir0 block mismatch.");
                this->PrintError(
                    "  maxDiff=%.8g thresh=%.8g (maxRef=%.8g)", maxDiff, thresh, maxRef
                );
                success = false;
                break;
            }
        }

        offset += nCells;
    }

    delete[] f;

    delete uqh;
    delete gridF;
    delete gridK;
    delete gridP;

    return success;
}
