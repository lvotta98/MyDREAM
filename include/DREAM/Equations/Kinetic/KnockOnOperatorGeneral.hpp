#ifndef _DREAM_EQUATIONS_KNOCK_ON_OPERATOR_GENERAL_HPP
#define _DREAM_EQUATIONS_KNOCK_ON_OPERATOR_GENERAL_HPP

#include "FVM/Equation/EquationTerm.hpp"
#include "FVM/Grid/Grid.hpp"
#include "FVM/config.h"

namespace DREAM {
class KnockOnOperatorGeneral : public FVM::EquationTerm {
   private:
    // quantities for building the final equation term
    FVM::Grid *grid_primary;
    FVM::UnknownQuantityHandler *unknowns;
    len_t id_ntot;
    len_t id_Efield;
    len_t id_f_primary;

    real_t pCutoff;

    // tabulated values of size Nr x n_xi_stars_tabulate x (Nxi*Nxi)
    len_t n_xi_stars_tabulate;
    real_t xiStar_min = 0.0;
    real_t xiStar_max = 1.0;
    real_t dXiStar = 1.0;
    real_t ***deltaTable = nullptr;
    real_t *xiStarsTab = nullptr;
    
    real_t *sourceVector = nullptr;
    real_t *mollerSMatrix = nullptr;

    real_t t_source_rebuilt;
    void Allocate();
    void Deallocate();
    void TabulateDeltaMatrixOnXiStarGrid();
    void SetMollerSMatrix(real_t*);
    void SetSourceVector(const real_t *f_primary);

   public:
    KnockOnOperatorGeneral(
        FVM::Grid *grid_knockon, FVM::Grid *grid_primary, FVM::UnknownQuantityHandler *unknowns,
        len_t id_f_primary, real_t p_cutoff, len_t n_xi_stars_tabulate = 50,
        len_t n_points_integral = 80
    );
    ~KnockOnOperatorGeneral();

    virtual void Rebuild(const real_t, const real_t, FVM::UnknownQuantityHandler*) override;
    virtual bool GridRebuilt() override;
    virtual void SetVectorElements(real_t*, const real_t*) override;
    virtual bool SetJacobianBlock(const len_t uqtyId, const len_t derivId, FVM::Matrix *jac, const real_t* x) override;
    virtual len_t GetNumberOfNonZerosPerRow() const override 
        { return 1; }

    
    len_t n_points_integral;

    // The below are mainly used in unit testing 
    real_t EvaluateDeltaMatrixElement(len_t ir, len_t i, len_t k, len_t j, len_t l);
    const real_t *GetSourceVector() const { return sourceVector; }
};
}  // namespace DREAM

#endif /*_DREAM_EQUATIONS_KNOCK_ON_OPERATOR_GENERAL_HPP*/
