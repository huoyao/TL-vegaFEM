#include "cula_sparse.h"
#include "cula_sparse_legacy.h"
#include "cuda_runtime.h"
#include <cstdlib>
#include <cstdio>
using namespace std;

#define MAT_CSR
//#define MAT_COO

#define SOLVER_CG
//#define SOLVER_BICG_STAB
//#define SOLVER_BICG

#define PRECONDITION_JACOBI
//#define PRECONDITION_ILU
//#define PRECONDITION_FAINV

void solverCulaCG(int *colInd,int *rowInd,double *a,double *x,double *b,int n,int nnz)
{
  // create library handle
  culaSparseHandle handle;
  culaSparseCreate(&handle);
  // create execution plan
  culaSparsePlan plan;
  culaSparseCreatePlan(handle, &plan);
  // initialize CUDA platform options
  culaSparseCudaDeviceOptions platformOpts;
  culaSparseStatus status;
  status=culaSparseCudaDeviceOptionsInit(handle,&platformOpts);
  // set CUDA platform
  status = culaSparseSetCudaDevicePlatform(handle,plan, &platformOpts);
  /************************************************************************/
  /* data formate                                                         */
  /************************************************************************/
#ifdef MAT_COO
  //for coo
  // initialize format options
  culaSparseCooOptions coo_formatOpts;
  culaSparseCooOptionsInit(handle, &coo_formatOpts);
  // change the indexing on host data to 0-based (default style)
  coo_formatOpts.indexing = 0;
  // associate coo data with the plan
  culaSparseSetDcooData(handle, plan, &coo_formatOpts, n, nnz, a, rowInd, colInd, x, b);
#endif

#ifdef MAT_CSR
  //for csr
  // initialize format options
  culaSparseCsrOptions csr_formatOpts;
  culaSparseCsrOptionsInit(handle, &csr_formatOpts);
  // change the indexing on host data to 0-based (default style)
  csr_formatOpts.indexing = 0;
  // associate coo data with the plan
  culaSparseSetDcsrData(handle, plan, &csr_formatOpts, n, nnz, a, rowInd, colInd, x, b);
#endif

  /************************************************************************/
  /* CG Solver                                                            */
  /************************************************************************/
#ifdef SOLVER_CG
  // create default cg options
  culaSparseCgOptions solverOpts;
  //culaSparseCudaDeviceOptions solverOpts;
  status=culaSparseCgOptionsInit(handle,&solverOpts);
  // associate cg solver with the plan
  status = culaSparseSetCgSolver(handle, plan, &solverOpts);
#endif

  /************************************************************************/
  /* BICG stab Solver                                                     */
  /************************************************************************/
#ifdef SOLVER_BICG_STAB
  // create default BICG stab options
  culaSparseBicgstabOptions solverOpts;
  //culaSparseCudaDeviceOptions solverOpts;
  status=culaSparseBicgstabOptionsInit(handle,&solverOpts);
  // associate cg solver with the plan
  status = culaSparseSetBicgstabSolver(handle, plan, &solverOpts);
#endif

  /************************************************************************/
  /* BICG Solver                                                     */
  /************************************************************************/
#ifdef SOLVER_BICG
  // create default BICG options
  culaSparseBicgOptions solverOpts;
  //culaSparseCudaDeviceOptions solverOpts;
  status=culaSparseBicgOptionsInit(handle,&solverOpts);
  // associate cg solver with the plan
  status = culaSparseSetBicgSolver(handle, plan, &solverOpts);
#endif

  /************************************************************************/
  /* Jacobi preconditioner                                                */
  /************************************************************************/
#ifdef PRECONDITION_JACOBI
  // initialize jacobi options
  culaSparseJacobiOptions precondOpts;
  status = culaSparseJacobiOptionsInit(handle, &precondOpts);
  // associate plan with jacobi preconditioner
  status = culaSparseSetJacobiPreconditioner(handle, plan, &precondOpts);
#endif

  /************************************************************************/
  /* Ilu preconditioner                                                   */
  /************************************************************************/
#ifdef PRECONDITION_ILU
  //initialize ilu options
  culaSparseIlu0Options precondOptsILU;
  status=culaSparseIlu0OptionsInit(handle,&precondOptsILU);
  //associate plan with ILU preconditioner
  status=culaSparseSetIlu0Preconditioner(handle,plan,&precondOptsILU);
#endif

  /************************************************************************/
  /* Fainv preconditioner                                                 */
  /************************************************************************/
#ifdef PRECONDITION_FAINV
  //initialize fainv options [failed]
  culaSparseFainvOptions precondOptsILU;
  status=culaSparseFainvOptionsInit(handle,&precondOptsILU);
  //associate plan with fainv preconditioner
  status=culaSparseSetFainvPreconditioner(handle,plan,&precondOptsILU);
#endif

  // create configuration structure
  culaSparseConfig config;
  // initialize values
  culaSparseConfigInit(handle,&config);
  // configure specific parameters
  config.relativeTolerance = 1e-30;
  config.absoluteTolerance=1e-30;
  config.maxIterations = 5000;
  config.maxRuntime = 1000;
  // information returned by the solver
  culaSparseResult results;
  // execute plan
  culaSparseExecutePlan(handle, plan, &config, &results);
  // allocate result string buffer
  const int bufferSize = 256;
  char buffer[bufferSize];
  // fill buffer with result string
  culaSparseGetResultString( handle, &results, buffer, bufferSize );
  // print result string to screen
  printf( "%s\n", buffer );
  // cleanup
  culaSparseDestroyPlan(plan);
  culaSparseDestroy(handle);
  printf("do something\n");
}
