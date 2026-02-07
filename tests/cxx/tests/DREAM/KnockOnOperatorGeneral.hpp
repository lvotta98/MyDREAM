#ifndef _DREAMTESTS_DREAM_KNOCK_ON_OPERATOR_GENERAL_HPP
#define _DREAMTESTS_DREAM_KNOCK_ON_OPERATOR_GENERAL_HPP

#include <string>

#include "UnitTest.hpp"

namespace DREAMTESTS::_DREAM {

/**
 * Unit/integration tests for DREAM::KnockOnOperatorGeneral.
 *
 * These tests validate invariants of the assembled source term as exposed through the
 * EquationTerm interface (Rebuild(), SetVectorElements(), SetJacobianBlock()).
 *
 * Covered properties:
 *  - Pitch-angle delta-column normalization (via EvaluateDeltaMatrixElement()).
 *  - Linearity in f_primary (tested on the assembled contribution).
 *  - Jacobian consistency w.r.t. n_tot (finite difference vs SetJacobianBlock()).
 *  - Global production identity combining Delta normalization and Møller S conservation.
 *  - Explicit-time caching behavior (Rebuild should only update when time changes).
 *  - Non-negativity for f_primary >= 0 and n_tot >= 0 (within numerical noise).
 *  - Radius locality: only cells belonging to radius ir depend on n_tot[ir].
 */
class KnockOnOperatorGeneral : public UnitTest {
   public:
    KnockOnOperatorGeneral(const std::string &s) : UnitTest(s) {}

    // Interpolated delta columns preserve normalization:
    //   sum_j dxi_j * Delta_interp(j,l; xiStar(p_i,p1_k)) ~= 1
    bool CheckKOG_DeltaInterpolationConservation();

    // Check that sum_i dp_i * S_ik = Stot_k
    bool CheckKOG_MollerKernelConservation();

    // Linearity of assembled source term in f_primary:
    //   S(fA+fB) ~= S(fA) + S(fB)
    bool CheckKOG_LinearityInFPrimary();

    // Finite-difference consistency test for Jacobian w.r.t. n_tot:
    // compare SetJacobianBlock() column against FD derivative of SetVectorElements()
    bool CheckKOG_JacobianFiniteDifferenceNt();

    // Integration test combining Delta normalization and Moller S conservation:
    //   ∑_{i,j} dp dxi Vp * source(i,j) == ∑_{k,l} dp1 dxi1 Vp1 f(k,l) * [∑_i dp S_{ik}]
    bool CheckKOG_GlobalProductionIdentity();

    // Cross-grid test: primaries on runaway grid sourcing hot grid
    bool CheckKOG_HotRunawayGlobalProductionIdentity();

    bool CheckKOG_TimeCachingRegression();

    bool CheckKOG_NonNegativity();

    bool CheckKOG_RadiusLocality();

    virtual bool Run(bool) override;
};

}  // namespace DREAMTESTS::_DREAM

#endif /*_DREAMTESTS_DREAM_KNOCK_ON_OPERATOR_GENERAL_HPP*/
