#ifndef _DREAMTESTS_DREAM_KNOCK_ON_OPERATOR_GENERAL_HPP
#define _DREAMTESTS_DREAM_KNOCK_ON_OPERATOR_GENERAL_HPP

#include <string>

#include "UnitTest.hpp"

namespace DREAMTESTS::_DREAM {

/**
 * Unit/integration tests for DREAM::KnockOnOperatorGeneral.
 *
 * These tests validate invariants of the assembled source operator:
 *  - Interpolated delta-column conservation (normalization).
 *  - Linearity in f_primary.
 *  - Jacobian consistency w.r.t. n_tot (finite difference).
 *  - Global production identity combining Delta normalization and Moller S conservation.
 */
class KnockOnOperatorGeneral : public UnitTest {
   public:
    KnockOnOperatorGeneral(const std::string &s) : UnitTest(s) {}

    // Interpolated delta columns preserve normalization:
    //   sum_j dxi_j * Delta_interp(j,l; xiStar(p_i,p1_k)) ~= 1
    bool CheckKOG_DeltaInterpolationConservation();

    // Linearity of assembled source vector in f_primary:
    //   S(fA+fB) ~= S(fA) + S(fB)
    bool CheckKOG_SourceVectorLinearity();

    // Finite-difference consistency test for Jacobian w.r.t. n_tot:
    // compare SetJacobianBlock() column against FD derivative of SetVectorElements()
    bool CheckKOG_JacobianFiniteDifferenceNt();

    // Integration test combining Delta normalization and Moller S conservation:
    //   ∑_{i,j} dp dxi Vp * source(i,j) == ∑_{k,l} dp1 dxi1 Vp1 f(k,l) * [∑_i dp S_{ik}]
    bool CheckKOG_GlobalProductionIdentity();

    virtual bool Run(bool) override;
};

}  // namespace DREAMTESTS::_DREAM

#endif /*_DREAMTESTS_DREAM_KNOCK_ON_OPERATOR_GENERAL_HPP*/
