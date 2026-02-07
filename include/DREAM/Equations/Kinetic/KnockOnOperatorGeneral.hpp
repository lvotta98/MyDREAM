#ifndef _DREAM_EQUATIONS_KNOCK_ON_OPERATOR_GENERAL_HPP
#define _DREAM_EQUATIONS_KNOCK_ON_OPERATOR_GENERAL_HPP

#include "FVM/Equation/EquationTerm.hpp"
#include "FVM/Grid/Grid.hpp"
#include "FVM/config.h"

namespace DREAM {

/**
 * Knock-on source term from large-angle binary (Møller) collisions.
 *
 * This equation term represents a kinetic source proportional to the total
 * target density n_tot(r). The term is treated explicitly in time: the source
 * is rebuilt only when the time level changes (not on each Newton iteration).
 *
 * Implementation notes:
 *  - Relies on tabulated pitch-angle delta kernels from KnockOnUtilities.
 *  - Assumes p-\xi momentum grids are identical across radii (validated).
 */
class KnockOnOperatorGeneral : public FVM::EquationTerm {
    enum XiClamp { Interp = 0, ClampLow = 1, ClampHigh = 2 };

    struct XiStarInterp {
        len_t m0;       // valid if clamp == Interp
        real_t w1;      // valid if clamp == Interp
        XiClamp clamp;  // interpolation or endpoint clamping
    };

   private:
    FVM::Grid *gridPrimary;
    FVM::UnknownQuantityHandler *unknowns;

    len_t id_ntot;
    len_t id_f_primary;

    real_t pCutoff;
    real_t scaleFactor;
    len_t nXiStarsTabulate;
    len_t nPointsIntegral;

    // Reference grid in xi_star used for delta-kernel tabulation.
    real_t xiStarMin = 0.0;
    real_t xiStarMax = 1.0;
    real_t dXiStar = 1.0;
    real_t *xiStarsTab = nullptr;

    // Delta kernels tabulated on xi_star grid:
    //   deltaTable[ir][m][j + l*NxiK] = Delta_{j l}(xi_star[m])
    // where NxiK is knock-on pitch resolution, and l indexes the primary pitch.
    real_t ***deltaTable = nullptr;
    real_t **deltaTableStorage = nullptr;  // contiguous storage per radius

    // Møller momentum-space kernel S_{ik} integrated over outgoing p-cells.
    real_t *mollerSMatrix = nullptr;

    // Precomputed xi_star interpolation meta-data for each (i,k) pair.
    XiStarInterp **xiInterp = nullptr;

    // Assembled source vector (per momentum-space cell on the knock-on grid).
    real_t *sourceVector = nullptr;
    real_t t_source_rebuilt;

    // Scratch buffers reused in SetSourceVector().
    real_t *primaryWeights = nullptr;  // size: Np1P*NxiP, layout [k*NxiP + l]
    real_t *pitchAccum = nullptr;      // size: NxiK

    void ValidateInputParameters() const;
    void ValidateGridAssumptions() const;

    void AllocateAndBuildTables();
    void Deallocate();

    void TabulateDeltaMatrixOnXiStarGrid();
    void BuildMollerSMatrix();
    void BuildXiStarInterp();
    void AllocateScratchBuffers();

    void BuildPrimaryWeights(len_t ir, const real_t *f_primary_ir, real_t *W_k_l) const;
    void SetSourceVector(const real_t *f_primary);

    void SelectDeltaPlanes(
        len_t ir, len_t i, len_t k, const real_t *&D0, const real_t *&D1, real_t &w0, real_t &w1
    ) const;

   public:
    KnockOnOperatorGeneral(
        FVM::Grid *grid_knockon, FVM::Grid *grid_primary, FVM::UnknownQuantityHandler *unknowns,
        len_t id_f_primary, real_t p_cutoff, real_t scaleFactor = 1.0,
        len_t n_xi_stars_tabulate = 100, len_t n_points_integral = 80
    );
    ~KnockOnOperatorGeneral();

    virtual void Rebuild(const real_t, const real_t, FVM::UnknownQuantityHandler *) override;
    virtual bool GridRebuilt() override;
    virtual void SetVectorElements(real_t *, const real_t *) override;
    virtual bool SetJacobianBlock(
        const len_t uqtyId, const len_t derivId, FVM::Matrix *jac, const real_t *x
    ) override;
    virtual len_t GetNumberOfNonZerosPerRow() const override { return 1; }

    void AccumulateAngleKernel(
        len_t ir, len_t i, len_t k, const real_t *W_l, real_t Sik, real_t *outPitch_j
    ) const;
    inline real_t MollerDifferentialCrossSection(len_t ir, len_t i, len_t k) const {
        // S(i,k) = differential cross section integrated over outgoing control volume i.
        return mollerSMatrix[i * gridPrimary->GetNp1(ir) + k];
    }
};

}  // namespace DREAM

#endif /*_DREAM_EQUATIONS_KNOCK_ON_OPERATOR_GENERAL_HPP*/
