// written by André Betz 
// http://www.andrebetz.de


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LINEAR                                0           /* linear kernel type */
#define POLY                                  1           /* polynoial kernel type */
#define RBF                                   2           /* rbf kernel type */
#define SIGMOID                               3           /* sigmoid kernel type */
#define PROGRESS                              1
#define NO_PROGRESS_MAXITER                   2
#define NO_PROGRESS_PRIMAL_OPTIMAL            3
#define NO_PROGRESS_DUAL_OPTIMAL              4
#define NAN_SOLUTION                          5
#define PROGRESS_PRIMAL_OPTIMAL               6
#define PROGRESS_MAXITER_AND_PRIMAL_OPTIMAL   7
#define PROGRESS_DUAL_AND_PRIMAL_OPTIMAL      8
#define PROGRESS_DUAL_NOT_PRIMAL_OPTIMAL      9
#define PROGRESS_MAXITER_NOT_PRIMAL_OPTIMAL   10
#define ONLY_ONE_VARIABLE                     11
#define PRIMAL_OPTIMAL                        1
#define DUAL_OPTIMAL                          2
#define MAXITER_EXCEEDED                      3
#define DEF_PRECISION                         1E-5
#define DEF_MAX_ITERATIONS                    200
#define DEF_LINDEP_SENSITIVITY                1E-8
#define EPSILON_HIDEO                         1E-20
#define CFLOAT float
#define FVAL   float

typedef struct doc 
{
  long    docnum;
  FVAL    *words;
} DOC;

typedef struct learn_parm 
{
  long   svm_c_steps;          /* do so many steps for finding optimal C */
  double svm_c_factor;         /* increase C by this factor every step */
  double svm_c;
  double svm_costratio;
  double svm_costratio_unlab;
  double svm_unlabbound;
  double *svm_cost;            /* individual upper bounds for each var */
  double transduction_posratio;/* fraction of unlabeled examples to be classified as positives */
  long   remove_inconsistent;
  long   biased_hyperplane;
  long   skip_final_opt_check;
  long   svm_maxqpsize;
  long   svm_iter_to_shrink;
  double epsilon_crit;         /* tolerable error for distances used in stopping criterion */
  double epsilon_a;            /* tolerable error on alphas at bounds */
  double epsilon_shrink;       /* how much a multiplier should be above zero for shrinking */
  double epsilon_const;        /* tolerable error on eq-constraint */
  double opt_precision;        /* precision of solver, set to 1e-21 */
} LEARN_PARM;

typedef struct kernel_parm 
{
  long    kernel_type;
  double  gamma;
  double  coef_lin;
  double  coef_const;
} KERNEL_PARM;

typedef struct model 
{
  long    sv_num;       
  long    at_upper_bound;
  double  b;
  DOC     **supvec;
  double  *alpha;
  long    *index;       /* index from docnum to position in model */
  double  *lin_weights; /* weights for linear case using folding */
} MODEL;

typedef struct quadratic_program 
{
  long   opt_n;            /* number of variables */
  long   opt_m;            /* number of linear equality constraints */
  double *opt_ce,*opt_ce0; /* linear equality constraints */
  double *opt_g;           /* hessian of objective */
  double *opt_g0;          /* linear part of objective */
  double *opt_xinit;       /* initial value for variables */
  double *opt_low,*opt_up; /* box constraints */
} QP;

typedef struct kernel_cache 
{
  long   *index;  /* cache some kernel evalutations */
  CFLOAT *buffer; /* to improve speed */
  long   *invindex;
  long   *active2totdoc;
  long   *totdoc2active;
  long   *lru;
  long   *occu;
  long   elems;
  long   max_elems;
  long   time;
  long   activenum;
  long   buffsize;
} KERNEL_CACHE;

class svm
{
public:
  svm(long lVec,long lDim,long lKType,double dVar);
  ~svm();
  void   Add(unsigned char* pucVec,long lClass,long lVecNum);
  void   GetModel(double* pfSupVec,double* pfAlpha,FVAL* Error);
  void   SetModel(long lSupAnz,double* pfSupMat,double* pfAlphaVec,FVAL Error);
  long   Train();
  long   Classify(unsigned char* pucVec);
protected:
  DOC          *docs;
  FVAL         *m_xTestDoc;
  long         *label;
  long         totwords;
  long         totdoc;
  long         kernel_cache_size;
  KERNEL_CACHE kernel_cache;
  LEARN_PARM   learn_parm;
  KERNEL_PARM  kernel_parm;
  MODEL        model;
  QP           qp;
  double       *buffer;
  long         *nonoptimal;
  int          nx;
  double       *primal;
  double       *dual;
  long         precision_violations;
  double       opt_precision;
  long         maxiter;
  double       lindep_sensitivity;
  long         count_nppo;

  double *optimize_qp(double *,long,double *);
  CFLOAT kernel(FVAL *,FVAL *); 
  double sprod_ss(FVAL *,FVAL *);
  void   add_vector_ns(double *,FVAL *, double);
  double sprod_ns(double *,FVAL *);
  long   max(long, long);
  double compute_objective_function(double *, double *, long *, long *);
  void   add_to_index(long *,long);
  long   compute_index(long *,long *);
  void   compute_matrices_for_optimization(long *unlabeled,long *chosen,long *active2dnum,long *key,double *a,double *lin,long varnum,CFLOAT *aicache);
  long   calculate_svm_model(long *unlabeled,double *lin,double *a,double *a_old,long *working2dnum);
  long   check_optimality(long *unlabeled,double *a,double *lin,double *maxdiff,double epsilon_crit_org, long *misclassified,long *inconsistent,long *active2dnum,long *last_suboptimal_at,long iteration);
  long   identify_inconsistent(double *a,long *unlabeled,long *inconsistentnum,long *inconsistent);
  long   incorporate_unlabeled_examples(long *inconsistent,long *unlabeled,double *a,double *lin,double *selcrit,long *select,long *key,long transductcycle);
  void   update_linear_component(long *active2dnum,double *a, double *a_old,long *working2dnum,double *lin,CFLOAT *aicache,double *weights);
  long   select_next_qp_subproblem_grad      (long *unlabeled,double *a,double *lin,long qp_size,long *inconsistent,long *active2dnum,long *working2dnum,double *selcrit,long *select,long *key,long *chosen);
  long   select_next_qp_subproblem_grad_cache(long *unlabeled,double *a,double *lin,long qp_size,long *inconsistent,long *active2dnum,long *working2dnum,double *selcrit,long *select,long *key,long *chosen);
  void   select_top_n(double *, long, long *, long);
  long   shrink_problem(long *last_suboptimal_at,long *active,long *inactive_since,long *active2dnum,long *deactnum,long iteration,long minshrink,double *a,double **a_history,long *inconsistent);
  void   reactivate_inactive_examples(long *unlabeled,double *a,double **a_history,double *lin,long iteration,long *inconsistent,long *active,long *inactive_since,long deactnum,CFLOAT *aicache,double *weights,double *maxdiff);
  void   get_kernel_row(long,long *,CFLOAT *);
  void   cache_multiple_kernel_rows(long *,long);
  void   kernel_cache_shrink(long,long *);
  void   kernel_cache_init();
  void   kernel_cache_cleanup();
  long   kernel_cache_malloc();
  long   kernel_cache_free_lru();
  CFLOAT *kernel_cache_clean_and_malloc(long);
  long   kernel_cache_touch(long);
  int    optimize_hildreth_despo(long,long,double,double,long,double,double *,double *,double *,double *,double *,double *,double *,double *,double*,long *,double *);
  int    solve_dual(long,long,double,double,long,double *,double *,double*,double *,double *,double *,double *,double *,double *,double *,double*,double *,double *);
  void   linvert_matrix(double *, long, double *, double, long *);
  void   lcopy_matrix(double *, long, double *);
  void   lswitchrk_matrix(double *, long, long, long);
  double calculate_qp_objective(long, double *, double *, double *);
}; 


svm::svm(long lVec,long lDim,long lKType,double dVar)
{
  long lCount;

  nx                               = 400;
  primal                           = 0;
  dual                             = 0;
  precision_violations             = 0;
  opt_precision                    = DEF_PRECISION;
  maxiter                          = DEF_MAX_ITERATIONS;
  lindep_sensitivity               = DEF_LINDEP_SENSITIVITY;
  count_nppo                       = 0;
  kernel_cache_size                = 40;
  learn_parm.biased_hyperplane     = 1;
  learn_parm.remove_inconsistent   = 0;
  learn_parm.skip_final_opt_check  = 0;
  learn_parm.svm_maxqpsize         = 10;
  learn_parm.svm_iter_to_shrink    = 100;
  learn_parm.svm_c                 = 1000.0;
  learn_parm.transduction_posratio =-1.0;
  learn_parm.svm_costratio         = 1.0;
  learn_parm.svm_costratio_unlab   = 1.0;
  learn_parm.svm_unlabbound        = 0.00001;
  learn_parm.epsilon_crit          = 0.001;
  learn_parm.epsilon_a             = 1E-12;
  learn_parm.epsilon_shrink        = 2;
  kernel_parm.kernel_type          = lKType;
  kernel_parm.gamma                = dVar;
  kernel_parm.coef_lin             = 1.0;
  kernel_parm.coef_const           = 1.0;
  model.sv_num                     = 0;
  model.at_upper_bound             = 0;
  model.b                          = 0;

  totdoc   = lVec;
  totwords = lDim;

  docs  = new DOC[totdoc];
  label = new long[totdoc];
  m_xTestDoc = new FVAL[totwords];

  for(lCount=0;lCount<totdoc;lCount++)
  {
    docs[lCount].docnum  = lCount;
    docs[lCount].words   = new FVAL[totwords];
  }
  
  qp.opt_ce            = new double[learn_parm.svm_maxqpsize];
  qp.opt_ce0           = new double;
  qp.opt_g             = new double[learn_parm.svm_maxqpsize*learn_parm.svm_maxqpsize];
  qp.opt_g0            = new double[learn_parm.svm_maxqpsize];
  qp.opt_xinit         = new double[learn_parm.svm_maxqpsize];
  qp.opt_low           = new double[learn_parm.svm_maxqpsize];
  qp.opt_up            = new double[learn_parm.svm_maxqpsize];
}

svm::~svm()
{
  delete m_xTestDoc;
  if(docs)  delete docs;
  if(label) delete label;
  delete model.supvec;
  delete model.alpha;
  delete model.index;
  delete qp.opt_ce;
  delete qp.opt_ce0;
  delete qp.opt_g;
  delete qp.opt_g0;
  delete qp.opt_xinit;
  delete qp.opt_low;
  delete qp.opt_up;
}

void svm::Add(unsigned char* pucVec,long lClass,long lVecNum)
{
  long  lCount;
  label[lVecNum] = lClass;
  for(lCount=0;lCount<totwords;lCount++) docs[lVecNum].words[lCount] = (FVAL)pucVec[lCount];
}

long svm::Classify(unsigned char* pucVec)
{ 
  double dDist = 0.0;
  long   lCount;
  long   lRecog = 0;

  for(lCount=0;lCount<totwords;lCount++) m_xTestDoc[lCount] = (FVAL)pucVec[lCount];
  for(lCount=1;lCount<model.sv_num;lCount++) dDist += kernel(model.supvec[lCount]->words,m_xTestDoc) * model.alpha[lCount];

  dDist -= model.b;

  if(dDist>0) lRecog =  1;
  else        lRecog = -1;

  return lRecog;
}

void svm::GetModel(double* pfSupVec,double* pfAlpha,FVAL* Error)
{
  long lCount1;
  long lCount2;

  (*Error)   = model.b;

  for(lCount1=1;lCount1<model.sv_num;lCount1++)
  {
    pfAlpha[lCount1-1] = model.alpha[lCount1];
    for(lCount2=0;lCount2<totwords;lCount2++) pfSupVec[(lCount1-1)*totwords+lCount2] = (double)(model.supvec[lCount1])->words[lCount2];
  }
}

void svm::SetModel(long lSupAnz,double* pfSupMat,double* pfAlphaVec,FVAL Error)
{
  long lCount1;
  long lCount2;

  model.sv_num = lSupAnz+1;
  model.b      = Error;
  model.supvec = new DOC*[model.sv_num];
  model.alpha  = new double[model.sv_num];

  for(lCount1=1;lCount1<model.sv_num;lCount1++)
  {
    model.alpha[lCount1]         = pfAlphaVec[lCount1-1]; 
    model.supvec[lCount1]        = new DOC;
    model.supvec[lCount1]->words = new FVAL[totwords];

    for(lCount2=0;lCount2<totwords;lCount2++) (model.supvec[lCount1])->words[lCount2] = pfSupMat[(lCount1-1)*totwords+lCount2];
  }
}

CFLOAT svm::kernel(FVAL* a,FVAL* b)
{
  switch(kernel_parm.kernel_type) 
  {
    case 0:  return((CFLOAT)sprod_ss(a,b)); 
    case 1:  return((CFLOAT)pow( kernel_parm.coef_lin * sprod_ss(a,b) + kernel_parm.coef_const,(double)kernel_parm.gamma)); 
    case 2:  return((CFLOAT)exp(-kernel_parm.gamma    * (sprod_ss(a,a) - 2 * sprod_ss(a,b) + sprod_ss(b,b))));
    case 3:  return((CFLOAT)tanh(kernel_parm.coef_lin * sprod_ss(a,b) + kernel_parm.coef_const)); 
    default: return((CFLOAT)1.0);
  }
}

long svm::Train()
{
  long   *inconsistent,*chosen,*key,i,j,jj,*last_suboptimal_at;
  long   inconsistentnum,choosenum,retrain,iteration,inactivenum;
  long   misclassified,supvecnum = 0,*active2dnum;
  long   *active,*working2dnum,*selexam;
  long   activenum,*inactive_since,deactnum;
  double eq;
  double maxdiff,*a_old,*lin,*a;
  double **a_history;
  long   transductcycle;
  long   *unlabeled,transduction;
  double epsilon_crit_org; 
  double *selcrit;  /* buffer for sorting */        
  CFLOAT *aicache;  /* buffer to keep one row of hessian */
  double *weights;  /* buffer for weight vector in linear case */

  kernel_cache_init();

  epsilon_crit_org = learn_parm.epsilon_crit; /* save org */
  maxdiff          = 1;
  if(kernel_parm.kernel_type==LINEAR) learn_parm.epsilon_crit = 2.0;

  inconsistent         = new long[totdoc];
  chosen               = new long[totdoc];
  unlabeled            = new long[totdoc];
  active               = new long[totdoc];
  inactive_since       = new long[totdoc];
  last_suboptimal_at   = new long[totdoc];
  key                  = new long[totdoc+11]; 
  selcrit              = new double[totdoc];
  selexam              = new long[totdoc];
  a                    = new double[totdoc];
  a_old                = new double[totdoc];
  a_history            = new double*[10000];
  lin                  = new double[totdoc];
  learn_parm.svm_cost  = new double[totdoc];
  aicache              = new CFLOAT[totdoc];
  working2dnum         = new long[totdoc+11];
  active2dnum          = new long[totdoc+11];
  weights              = new double[totwords+1];
  model.supvec         = new DOC*[totdoc+2];
  model.alpha          = new double[totdoc+2];
  model.index          = new long[totdoc+2];

  model.supvec[0]       = 0;  /* element 0 reserved and empty for now */
  model.alpha[0]        = 0;
  model.sv_num          = 1;
  activenum             = totdoc;
  choosenum             = 0;
  inactivenum           = 0;
  deactnum              = 0;
  inconsistentnum       = 0;
  transductcycle        = 0;
  transduction          = 0;
  retrain               = 1;

  for(i=0;i<totdoc;i++) 
  { 
    /* various inits */
    inconsistent[i] = 0;
    chosen[i]       = 0;
    active[i]       = 1;
    a[i]            = 0;
    a_old[i]        = 0;
    lin[i]          = 0;
    last_suboptimal_at[i] = 1;
    unlabeled[i]          = 0;
    if(label[i] == 0) 
    {
      unlabeled[i] = 1;
      transduction = 1;
    }
    if(label[i] > 0) 
    {
      learn_parm.svm_cost[i]=learn_parm.svm_c*learn_parm.svm_costratio*fabs((double)label[i]);
      label[i]=1;
    }
    else if(label[i] < 0) 
    {
      learn_parm.svm_cost[i]=learn_parm.svm_c*fabs((double)label[i]);
      label[i]=-1;
    }
    else 
    {
      learn_parm.svm_cost[i]=0;
    }
  }
  activenum = compute_index(active,active2dnum);
  working2dnum[0] = -1;

  /* repeat this loop until we have convergence */
  for(iteration=1;retrain;iteration++) 
  {
    if(kernel_parm.kernel_type != LINEAR) kernel_cache.time=iteration;  /* for lru cache */

    printf("."); fflush(stdout);

    choosenum=0;
    for(jj=0;(j=working2dnum[jj])>=0;jj++) chosen[j]=0; 
    working2dnum[0] = -1;

    if(retrain == 2) 
    {
      for(i=0;i<totdoc;i++) 
      {
        /* set inconsistent examples to zero (-i 1) */
	if(inconsistent[i] && (a[i] != 0.0)) 
        {
	  chosen[i]=1;
	  choosenum++;
	  a[i]=0;
	}
      }

      if(learn_parm.biased_hyperplane) 
      {
	eq=0;
	for(i=0;i<totdoc;i++) 
        { 
          /* make sure we fulfill equality constraint */
	  eq+=a[i]*label[i];
	}

	for(i=0;(i<totdoc) && (fabs(eq) > learn_parm.epsilon_a);i++) 
        {
	  if(eq*label[i] > 0) 
          {
	    chosen[i]=1;
	    choosenum++;
	    if((eq*label[i]) > a[i]) 
            {
	      eq-=(a[i]*label[i]);
	      a[i]=0;
	    }
	    else 
            {
	      a[i]-=(eq*label[i]);
	      eq=0;
	    }
	  }
	}

      }
      compute_index(chosen,working2dnum);
    }
    else 
    {
      /* select working set according to steepest gradient */ 
      if((learn_parm.svm_maxqpsize>=4) && (kernel_parm.kernel_type != LINEAR)) 
      {
	/* select part of the working set from cache */
	choosenum+=select_next_qp_subproblem_grad_cache(unlabeled,a,lin,(long)((learn_parm.svm_maxqpsize-choosenum)/2),
			       inconsistent,active2dnum,working2dnum,selcrit,selexam,key,chosen);
      }
      choosenum+=select_next_qp_subproblem_grad(unlabeled,a,lin,(long)(learn_parm.svm_maxqpsize-choosenum),
                               inconsistent,active2dnum,working2dnum,selcrit,selexam,key,chosen);
    }

    if(kernel_parm.kernel_type != LINEAR)cache_multiple_kernel_rows(working2dnum,choosenum); 
    
    if(retrain != 2) 
    {
      double *a_v;          

      /* optimize svm */
      compute_matrices_for_optimization(unlabeled,chosen,active2dnum,working2dnum,a,lin,choosenum,aicache);

      /* call the qp-subsolver */
      a_v = optimize_qp(&epsilon_crit_org,learn_parm.svm_maxqpsize,&(model.b));
     /* in case the optimizer gives us */
     /* the threshold for free. otherwise */
     /* b is calculated in calculate_model. */

      for(i=0;i<choosenum;i++) 
      {
        a[working2dnum[i]]=a_v[i];
      }
    }

    update_linear_component(active2dnum,a,a_old,working2dnum,lin,aicache,weights);
    supvecnum = calculate_svm_model(unlabeled,lin,a,a_old,working2dnum);

    /* The following computation of the objective function works only */
    /* relative to the active variables */

    for(jj=0;(i=working2dnum[jj])>=0;jj++) 
    {
      a_old[i]=a[i];
    }

    if(retrain == 2) 
    {
      for(i=0;(i<totdoc);i++) 
      {
	if(inconsistent[i] && unlabeled[i]) 
        {
	  inconsistent[i] = 0;
	  label[i]        = 0;
	}
      }
    }

    retrain=check_optimality(unlabeled,a,lin,&maxdiff,epsilon_crit_org,&misclassified,inconsistent,active2dnum,last_suboptimal_at,iteration);

    if((!retrain) && (inactivenum>0) && ((!learn_parm.skip_final_opt_check) || (kernel_parm.kernel_type == LINEAR))) 
    { 
      printf("\n");

      reactivate_inactive_examples(unlabeled,a,a_history,lin,iteration,inconsistent,active,inactive_since,deactnum,aicache,weights,&maxdiff);

      /* Update to new active variables. */
      activenum   = compute_index(active,active2dnum);
      inactivenum = totdoc-activenum;
      /* termination criterion */
      retrain = 0;
      if(maxdiff > learn_parm.epsilon_crit) retrain=1;
    }

    if((!retrain) && (learn_parm.epsilon_crit>maxdiff)) learn_parm.epsilon_crit = maxdiff;
    if((!retrain) && (learn_parm.epsilon_crit>epsilon_crit_org)) 
    {
      learn_parm.epsilon_crit /= 2.0;
      retrain = 1;
    }
    if(learn_parm.epsilon_crit<epsilon_crit_org) learn_parm.epsilon_crit=epsilon_crit_org;
    
    if((!retrain) && (transduction)) 
    {
      for(i=0;(i<totdoc);i++) 
      {
	active[i] = 1;
      }
      activenum=compute_index(active,active2dnum);
      inactivenum=0;
      retrain = incorporate_unlabeled_examples(inconsistent,unlabeled,a,lin,selcrit,selexam,key,transductcycle);

      epsilon_crit_org=learn_parm.epsilon_crit;
      if(kernel_parm.kernel_type == LINEAR) learn_parm.epsilon_crit=1; 
      transductcycle++;
    } 
    else if((iteration % 10) == 0) 
    {
      activenum=shrink_problem(last_suboptimal_at,active,inactive_since,active2dnum,&deactnum,
			       iteration,max((long)(activenum/10),100),a,a_history,inconsistent);
      inactivenum = totdoc - activenum;
      if((kernel_parm.kernel_type != LINEAR) && (supvecnum>kernel_cache.max_elems) && ((kernel_cache.activenum-activenum)>max((long)(activenum/10),500))) 
      {
	kernel_cache_shrink(max((long)(activenum/10),500),active); 
      }
    }

    if((!retrain) & learn_parm.remove_inconsistent) 
    {
      retrain = identify_inconsistent(a,unlabeled,&inconsistentnum,inconsistent);
    }
  } 

  delete inconsistent;
  delete chosen;
  delete unlabeled;
  delete active;
  delete inactive_since;
  delete last_suboptimal_at;
  delete key;
  delete selcrit;
  delete selexam;
  delete a;
  delete a_old;
  delete a_history;
  delete lin;
  delete learn_parm.svm_cost;
  delete aicache;
  delete working2dnum;
  delete active2dnum;
  delete qp.opt_ce;
  delete qp.opt_ce0;
  delete qp.opt_g;
  delete qp.opt_g0;
  delete qp.opt_xinit;
  delete qp.opt_low;
  delete qp.opt_up;
  delete weights;

  kernel_cache_cleanup();
  return model.sv_num-1;
}

void svm::add_to_index(long *index,long elem)
{
  register long i;
  for(i=0;index[i] != -1;i++);
  index[i]=elem;
  index[i+1]=-1;
}

long svm::compute_index(long *binfeature,long *index)
{               
  register long i,ii;

  ii=0;

  for(i=0;i<totdoc;i++) 
  {
    if(binfeature[i]) 
    {
      index[ii]=i;
      ii++;
    }
  }

  for(i=0;i<4;i++) 
  {
    index[ii+i]=-1;
  }
  return(ii);
}

void svm::compute_matrices_for_optimization(long *unlabeled,long *chosen,long *active2dnum,long *key,double *a,double *lin,long varnum,CFLOAT *aicache)
{
  register long ki,kj,i,j;
  register double kernel_temp;

  qp.opt_n=varnum;
  qp.opt_ce0[0]=0; /* compute the constant for quality constraint */

  for(j=1;j<model.sv_num;j++) 
  { 
    /* start at 1 */
    if(!chosen[(model.supvec[j])->docnum]) qp.opt_ce0[0]+=model.alpha[j];
  } 
  if(learn_parm.biased_hyperplane) qp.opt_m=1;
  else                             qp.opt_m=0;  /* eq-constraint will be ignored */

  /* init linear part of objective function */
  for(i=0;i<varnum;i++) qp.opt_g0[i]=lin[key[i]];

  for(i=0;i<varnum;i++) 
  {
    ki=key[i];

    /* Compute the matrix for equality constraints */
    qp.opt_ce[i]=label[ki];
    qp.opt_low[i]=0;
    qp.opt_up[i]=learn_parm.svm_cost[ki];

    kernel_temp=(double)kernel(docs[ki].words,docs[ki].words); 
    /* compute linear part of objective function */
    qp.opt_g0[i]-=(kernel_temp*a[ki]*(double)label[ki]); 
    /* compute quadratic part of objective function */
    qp.opt_g[varnum*i+i]=kernel_temp;
    for(j=i+1;j<varnum;j++) 
    {
      kj=key[j];
      kernel_temp=(double)kernel(docs[ki].words,docs[kj].words);
      /* compute linear part of objective function */
      qp.opt_g0[i]-=(kernel_temp*a[kj]*(double)label[kj]);
      qp.opt_g0[j]-=(kernel_temp*a[ki]*(double)label[ki]); 
      /* compute quadratic part of objective function */
      qp.opt_g[varnum*i+j]=(double)label[ki]*(double)label[kj]*kernel_temp;
      qp.opt_g[varnum*j+i]=(double)label[ki]*(double)label[kj]*kernel_temp;
    }
  }

  for(i=0;i<varnum;i++) 
  {
    /* assure starting at feasible point */
    qp.opt_xinit[i]=a[key[i]];
    /* set linear part of objective function */
    qp.opt_g0[i]=-1.0+qp.opt_g0[i]*(double)label[key[i]];    
  }
}

long svm::calculate_svm_model(long *unlabeled,double *lin,double *a,double *a_old,long *working2dnum)
{
  long i,ii,pos,b_calculated=0;
  double ex_c;

  if(!learn_parm.biased_hyperplane) 
  {
    model.b=0;
    b_calculated=1;
  }

  for(ii=0;(i=working2dnum[ii])>=0;ii++) 
  {
    if((a_old[i]>0) && (a[i]==0)) 
    { 
      /* remove from model */
      pos=model.index[i]; 
      model.index[i]=-1;
      model.sv_num--;
      model.supvec[pos] = model.supvec[model.sv_num];
      model.alpha[pos]  = model.alpha[model.sv_num];
      model.index[(model.supvec[pos])->docnum] = pos;
    }
    else if((a_old[i]==0) && (a[i]>0)) 
    { 
      /* add to model */
      model.supvec[model.sv_num] =& (docs[i]);
      model.alpha[model.sv_num]  = a[i]*(double)label[i];
      model.index[i]             = model.sv_num;
      model.sv_num++;
    }
    else if(a_old[i]==a[i]) 
    { /* nothing to do */
    }
    else 
    {  /* just update alpha */
      model.alpha[model.index[i]] = a[i]*(double)label[i];
    }
      
    ex_c=learn_parm.svm_cost[i] - learn_parm.epsilon_a;
    if((a_old[i]>=ex_c) && (a[i]<ex_c)) 
    { 
      model.at_upper_bound--;
    }
    else if((a_old[i]<ex_c) && (a[i]>=ex_c)) 
    { 
      model.at_upper_bound++;
    }

    if((!b_calculated) && (a[i]>learn_parm.epsilon_a) && (a[i]<ex_c)) 
    {   
       /* calculate b */
	model.b = (-(double)label[i]+lin[i]);
	b_calculated=1;
    }
  }      

  /* If there is no alpha in the working set not at bounds, then just use the model->b from the last iteration */

  return(model.sv_num-1); /* have to substract one, since element 0 is empty*/
}

long svm::check_optimality(long *unlabeled,double *a,double *lin,double *maxdiff,double epsilon_crit_org, long *misclassified,long *inconsistent,long *active2dnum,long *last_suboptimal_at,long iteration)
{
  long i,ii,retrain;
  double dist,ex_c;

  if(kernel_parm.kernel_type == LINEAR) 
  {  
    /* be optimistic */
    learn_parm.epsilon_shrink=-learn_parm.epsilon_crit+epsilon_crit_org;  
  }
  else 
  {  
    /* be conservative */
    learn_parm.epsilon_shrink=learn_parm.epsilon_shrink*0.7+(*maxdiff)*0.3; 
  }
  retrain=0;
  (*maxdiff)=0;
  (*misclassified)=0;
  for(ii=0;(i=active2dnum[ii])>=0;ii++) 
  {
    if((!inconsistent[i]) && label[i]) 
    {
      dist = (lin[i]-model.b)*(double)label[i];/* 'distance' from hyperplane*/
      ex_c = learn_parm.svm_cost[i] - learn_parm.epsilon_a;
      if(dist <= 0) 
      {       
	(*misclassified)++;  /* does not work due to deactivation of var */
      }
      if((a[i]>learn_parm.epsilon_a) && (dist > 1)) 
      {
	if((dist-1.0)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-1.0;
      }
      else if((a[i]<ex_c) && (dist < 1)) 
      {
	if((1.0-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=1.0-dist;
      }
      /* Count how long a variable was at lower/upper bound (and optimal).*/
      /* Variables, which were at the bound and optimal for a long */
      /* time are unlikely to become support vectors. In case our */
      /* cache is filled up, those variables are excluded to save */
      /* kernel evaluations. (See chapter 'Shrinking').*/ 
      if((a[i]>(learn_parm.epsilon_a)) && (a[i]<ex_c)) 
      { 
	last_suboptimal_at[i]=iteration;         /* not at bound */
      }
      else if((a[i]<=(learn_parm.epsilon_a)) && (dist < (1.0+learn_parm.epsilon_shrink))) 
      {
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
      else if((a[i]>=ex_c) && (dist > (1.0-learn_parm.epsilon_shrink)))  
      { 
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
    }   
  }
  /* termination criterion */
  if((!retrain) && ((*maxdiff) > learn_parm.epsilon_crit)) 
  {  
    retrain=1;
  }
  return(retrain);
}

long svm::identify_inconsistent(double *a,long *unlabeled,long *inconsistentnum,long *inconsistent)
{
  long i,retrain;

  /* Throw out examples with multipliers at upper bound. This */
  /* corresponds to the -i 1 option. */
  /* ATTENTION: this is just a heuristic for finding a close */
  /*            to minimum number of examples to exclude to */
  /*            make the problem separable with desired margin */
  retrain=0;
  for(i=0;i<totdoc;i++) 
  {
    if((!inconsistent[i]) && (!unlabeled[i]) && (a[i]>=(learn_parm.svm_cost[i]-learn_parm.epsilon_a))) 
    { 
	(*inconsistentnum)++;
	inconsistent[i]=1;  /* never choose again */
	retrain=2;          /* start over */
    }
  }
  return(retrain);
}

void svm::update_linear_component(long *active2dnum,double *a, double *a_old,long *working2dnum,double *lin,CFLOAT *aicache,double *weights)
{
  register long i,ii,j,jj;
  register double tec;

  if(kernel_parm.kernel_type==0) 
  { 
    /* special linear case */
    for(i=0;i<totwords;i++) weights[i] = 0;

    for(ii=0;(i=working2dnum[ii])>=0;ii++) 
    {
      if(a[i] != a_old[i]) 
      {
	add_vector_ns(weights,docs[i].words,((a[i]-a_old[i])*(double)label[i]));
      }
    }

    for(jj=0;(j=active2dnum[jj])>=0;jj++) 
    {
      lin[j]+=sprod_ns(weights,docs[j].words);
    }
  }
  else 
  {
    /* general case */
    for(jj=0;(i=working2dnum[jj])>=0;jj++) 
    {
      if(a[i] != a_old[i]) 
      {
	get_kernel_row(i,active2dnum,aicache);
		       
	for(ii=0;(j=active2dnum[ii])>=0;ii++) 
        {
	  tec=aicache[j];
	  lin[j]+=(((a[i]*tec)-(a_old[i]*tec))*(double)label[i]);
	}
      }
    }
  }
}

long svm::incorporate_unlabeled_examples(long *inconsistent,long *unlabeled,double *a,double *lin,double *selcrit,long *select,long *key,long transductcycle)
{
  long i,j,k,j1,j2,j3,j4,unsupaddnum1=0,unsupaddnum2=0;
  long pos,neg,upos,uneg,orgpos,orgneg,nolabel,newpos,newneg,allunlab;
  double dist,model_length,posratio,negratio;
  long check_every=2;
  double loss;
  static double switchsens=0.0,switchsensorg=0.0;
  double umin,umax,sumalpha;
  long imin=0,imax=0;
  static long switchnum=0;

  switchsens/=1.2;

  /* assumes that lin[] is up to date -> no inactive vars */

  orgpos=0;
  orgneg=0;
  newpos=0;
  newneg=0;
  nolabel=0;
  allunlab=0;
  for(i=0;i<totdoc;i++) 
  {
    if(!unlabeled[i]) 
    {
      if(label[i] > 0) 
      {
	orgpos++;
      }
      else  
      {
	orgneg++;
      }
    }
    else 
    {
      allunlab++;
      if(unlabeled[i]) 
      {
	if(label[i] > 0) 
        {
	  newpos++;
	}
	else if(label[i] < 0) 
        {
	  newneg++;
	}
      }
    }
    if(label[i]==0) 
    {
      nolabel++;
    }
  }

  if(learn_parm.transduction_posratio >= 0) 
  {
    posratio=learn_parm.transduction_posratio;
  }
  else 
  {
    posratio=(double)orgpos/(double)(orgpos+orgneg); /* use ratio of pos/neg */
  }                                                  /* in training data */
  negratio=1.0-posratio;

  learn_parm.svm_costratio=1.0;                     /* global */
  if(posratio>0) 
  {
    learn_parm.svm_costratio_unlab=negratio/posratio;
  }
  else 
  {
    learn_parm.svm_costratio_unlab=1.0;
  }
  
  pos=0;
  neg=0;
  upos=0;
  uneg=0;
  for(i=0;i<totdoc;i++) 
  {
    dist=(lin[i]-model.b);  /* 'distance' from hyperplane*/
    if(dist>0) 
    {
      pos++;
    }
    else 
    {
      neg++;
    }
    if(unlabeled[i]) 
    {
      if(dist>0) 
      {
	upos++;
      }
      else 
      {
	uneg++;
      }
    }
  }

  if(transductcycle == 0) 
  {
    j1=0; 
    j2=0;
    j4=0;
    for(i=0;i<totdoc;i++) 
    {
      dist=(lin[i]-model.b);  /* 'distance' from hyperplane*/
      if((label[i]==0) && (unlabeled[i])) 
      {
	selcrit[j4]=dist;
	key[j4]=i;
	j4++;
      }
    }
    unsupaddnum1=0;	
    unsupaddnum2=0;	
    select_top_n(selcrit,j4,select,(long)(allunlab*posratio+0.5));
    for(k=0;(k<(long)(allunlab*posratio+0.5));k++) 
    {
      i=key[select[k]];
      label[i]=1;
      unsupaddnum1++;	
      j1++;
    }
    for(i=0;i<totdoc;i++) 
    {
      if((label[i]==0) && (unlabeled[i])) 
      {
	label[i]=-1;
	j2++;
	unsupaddnum2++;
      }
    }
    for(i=0;i<totdoc;i++) 
    {  
      /* set upper bounds on vars */
      if(unlabeled[i]) 
      {
	if(label[i] == 1) 
        {
	  learn_parm.svm_cost[i]=learn_parm.svm_c * learn_parm.svm_costratio_unlab * learn_parm.svm_unlabbound;
	}
	else if(label[i] == -1) 
        {
	  learn_parm.svm_cost[i] = learn_parm.svm_c * learn_parm.svm_unlabbound;
	}
      }
    }

    return((long)3);
  }

  if((transductcycle % check_every) == 0) 
  {
    j1=0;
    j2=0;
    unsupaddnum1=0;
    unsupaddnum2=0;
    for(i=0;i<totdoc;i++) 
    {
      if((unlabeled[i] == 2)) 
      {
	unlabeled[i]=1;
	label[i]=1;
	j1++;
	unsupaddnum1++;
      }
      else if((unlabeled[i] == 3)) 
      {
	unlabeled[i]=1;
	label[i]=-1;
	j2++;
	unsupaddnum2++;
      }
    }
    for(i=0;i<totdoc;i++) 
    {  
      /* set upper bounds on vars */
      if(unlabeled[i]) 
      {
	if(label[i] == 1) 
        {
	  learn_parm.svm_cost[i] = learn_parm.svm_c * learn_parm.svm_costratio_unlab * learn_parm.svm_unlabbound;
	}
	else if(label[i] == -1) 
        {
	  learn_parm.svm_cost[i] = learn_parm.svm_c * learn_parm.svm_unlabbound;
	}
      }
    }

    if(learn_parm.svm_unlabbound == 1) 
    {
      learn_parm.epsilon_crit = 0.001; /* do the last run right */
    }
    else 
    {
      learn_parm.epsilon_crit=0.01; /* otherwise, no need to be so picky */
    }

    return((long)3);
  }
  else if(((transductcycle % check_every) < check_every)) 
  { 
    model_length=0;
    sumalpha=0;
    loss=0;
    for(i=0;i<totdoc;i++) 
    {
      model_length+=a[i]*label[i]*lin[i];
      sumalpha+=a[i];
      dist=(lin[i]-model.b);  /* 'distance' from hyperplane*/
      if((label[i]*dist)<(1.0-learn_parm.epsilon_crit)) 
      {
	loss+=(1.0-(label[i]*dist))*learn_parm.svm_cost[i]; 
      }
    }
    model_length=sqrt(model_length); 

    j1=0;
    j2=0;
    j3=0;
    j4=0;
    unsupaddnum1=0;	
    unsupaddnum2=0;	
    umin=99999;
    umax=-99999;
    j4=1;
    while(j4) 
    {
      umin=99999;
      umax=-99999;
      for(i=0;(i<totdoc);i++) 
      { 
	dist=(lin[i]-model.b);  
	if((label[i]>0) && (unlabeled[i]) && (!inconsistent[i]) && (dist<umin)) 
        {
	  umin=dist;
	  imin=i;
	}
	if((label[i]<0) && (unlabeled[i])  && (!inconsistent[i]) && (dist>umax)) 
        {
	  umax=dist;
	  imax=i;
	}
      }
      if((umin < (umax+switchsens-1E-4))) 
      {
	j1++;
	j2++;
	unsupaddnum1++;	
	unlabeled[imin]=3;
	inconsistent[imin]=1;
	unsupaddnum2++;	
	unlabeled[imax]=2;
	inconsistent[imax]=1;
      }
      else
	j4=0;
      j4=0;
    }

    for(j=0;(j<totdoc);j++) 
    {
      if(unlabeled[j] && (!inconsistent[j])) 
      {
	if(label[j]>0) 
        {
	  unlabeled[j]=2;
	}
	else if(label[j]<0) 
        {
	  unlabeled[j]=3;
	}
	/* inconsistent[j]=1; */
	j3++;
      }
    }

    switchnum+=unsupaddnum1+unsupaddnum2;

    if((!unsupaddnum1) && (!unsupaddnum2)) 
    {
      if((learn_parm.svm_unlabbound>=1) && ((newpos+newneg) == allunlab)) 
      {
	for(j=0;(j<totdoc);j++) 
        {
	  inconsistent[j]=0;
	  if(unlabeled[j]) unlabeled[j]=1;
	}

	return((long)0);
      }
      switchsens=switchsensorg;
      learn_parm.svm_unlabbound*=1.5;
      if(learn_parm.svm_unlabbound>1) 
      {
	learn_parm.svm_unlabbound=1;
      }
      model.at_upper_bound=0; /* since upper bound increased */
    }

    learn_parm.epsilon_crit=0.5; /* don't need to be so picky */

    for(i=0;i<totdoc;i++) 
    {  
      /* set upper bounds on vars */
      if(unlabeled[i]) 
      {
	if(label[i] == 1) 
        {
	  learn_parm.svm_cost[i]=learn_parm.svm_c * learn_parm.svm_costratio_unlab * learn_parm.svm_unlabbound;
	}
	else if(label[i] == -1) 
        {
	  learn_parm.svm_cost[i]=learn_parm.svm_c * learn_parm.svm_unlabbound;
	}
      }
    }

    return((long)2);
  }

  return((long)0); 
}

long svm::select_next_qp_subproblem_grad(long *unlabeled,double *a,double *lin,long qp_size,long *inconsistent,long *active2dnum,long *working2dnum,double *selcrit,long *select,long *key,long *chosen)
{
  long choosenum,i,j,k,activedoc,inum;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) 
  {
    s=-label[j];
    if((!((a[j]<=(0+learn_parm.epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm.svm_cost[j]-learn_parm.epsilon_a)) 
	     && (s>0)))
       && (!inconsistent[j]) 
       && (label[j])
       && (!chosen[j])) {
      selcrit[activedoc]=lin[j]-(double)label[j];
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) 
  {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(i); /* make sure it does not get kicked */
                                        /* out of cache */
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) 
  {
    s=label[j];
    if((!((a[j]<=(0+learn_parm.epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm.svm_cost[j]-learn_parm.epsilon_a)) 
	     && (s>0))) 
       && (!inconsistent[j]) 
       && (label[j])
       && (!chosen[j])) 
    {
      selcrit[activedoc]=(double)(label[j])-lin[j];
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) 
  {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(i); /* make sure it does not get kicked */
                                        /* out of cache */
  } 
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}

long svm::select_next_qp_subproblem_grad_cache(long *unlabeled,double *a,double *lin,long qp_size,long *inconsistent,long *active2dnum,long *working2dnum,double *selcrit,long *select,long *key,long *chosen)
{
  long choosenum,i,j,k,activedoc,inum;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) 
  {
    s=-label[j];
    if((kernel_cache.index[j]>=0)
       && (!((a[j]<=(0+learn_parm.epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm.svm_cost[j]-learn_parm.epsilon_a)) 
	     && (s>0)))
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j]))
      {
      selcrit[activedoc]=(double)label[j]*(-1.0+(double)label[j]*lin[j]);
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) 
  {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(i); /* make sure it does not get kicked */
                                        /* out of cache */
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) 
  {
    s=label[j];
    if((kernel_cache.index[j]>=0)
       && (!((a[j]<=(0+learn_parm.epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm.svm_cost[j]-learn_parm.epsilon_a)) 
	     && (s>0))) 
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j])) 
      {
      selcrit[activedoc]=-(double)(label[j]*(-1.0+(double)label[j]*lin[j]));
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) 
  {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(i); /* make sure it does not get kicked */
                                        /* out of cache */
  } 
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}

void svm::select_top_n(double *selcrit,long range,long *select,long n)
{
  register long i,j;

  for(i=0;(i<n) && (i<range);i++) { /* Initialize with the first n elements */
    for(j=i;j>=0;j--) {
      if((j>0) && (selcrit[select[j-1]]<selcrit[i])){
	select[j]=select[j-1];
      }
      else {
	select[j]=i;
	j=-1;
      }
    }
  }
  for(i=n;i<range;i++) {  
    if(selcrit[i]>selcrit[select[n-1]]) {
      for(j=n-1;j>=0;j--) {
	if((j>0) && (selcrit[select[j-1]]<selcrit[i])) {
	  select[j]=select[j-1];
	}
	else {
	  select[j]=i;
	  j=-1;
	}
      }
    }
  }
}      
      
long svm::shrink_problem(long *last_suboptimal_at,long *active,long *inactive_since,long *active2dnum,long *deactnum,long iteration,long minshrink,double *a,double **a_history,long *inconsistent)
{
  long i,ii,change,activenum;
  double *a_old;
  
  activenum=0;
  change=0;
  for(ii=0;active2dnum[ii]>=0;ii++) 
  {
    i=active2dnum[ii];
    activenum++;
    if(((iteration-last_suboptimal_at[i])>learn_parm.svm_iter_to_shrink) || (inconsistent[i])) 
    {
      change++;
    }
  }
  if(change>=minshrink) 
   { /* shrink only if sufficiently many candidates */
    /* Shrink problem by removing those variables which are */
    /* optimal at a bound for a minimum number of iterations */
    a_old = new double[totdoc]; 
    a_history[(*deactnum)]=a_old;
    for(i=0;i<totdoc;i++) 
    {
      a_old[i]=a[i];
    }
    change=0;
    for(ii=0;active2dnum[ii]>=0;ii++) 
    {
      i=active2dnum[ii];
      if((((iteration-last_suboptimal_at[i])>learn_parm.svm_iter_to_shrink) || (inconsistent[i]))) 
      {
	active[i]=0;
	inactive_since[i]=(*deactnum);
	change++;
      }
    }
    activenum=compute_index(active,active2dnum);
    (*deactnum)++;
  }
  return(activenum);
} 

void svm::reactivate_inactive_examples(long *unlabeled,double *a,double **a_history,double *lin,long iteration,long *inconsistent,
				  long *active,long *inactive_since,long deactnum,CFLOAT *aicache,double *weights,double *maxdiff)
{
  register long i,j,ii,jj,t,*changed2dnum,*inactive2dnum;
  long *changed,*inactive;
  register double kernel_val,*a_old,dist;
  double ex_c;

  changed       = new long[totdoc];
  changed2dnum  = new long[totdoc+11];
  inactive      = new long[totdoc];
  inactive2dnum = new long[totdoc+11];

  for(t=deactnum-1;(t>=0) && a_history[t];t--) 
  {
    a_old=a_history[t];    
    for(i=0;i<totdoc;i++) 
    {
      inactive[i]=((!active[i]) && (inactive_since[i] == t));
      changed[i]= (a[i] != a_old[i]);
    }
    compute_index(inactive,inactive2dnum);
    compute_index(changed,changed2dnum);

    if(kernel_parm.kernel_type == LINEAR) 
    { 
      /* special linear case */
      for(i=0;i<totwords;i++) weights[i]=0;

      for(ii=0;changed2dnum[ii]>=0;ii++) 
      {
	i=changed2dnum[ii];
	add_vector_ns(weights,docs[i].words,((a[i]-a_old[i])*(double)label[i]));
      }
      for(jj=0;(j=inactive2dnum[jj])>=0;jj++) 
      {
	lin[j]+=sprod_ns(weights,docs[j].words);
      }
    }
    else 
    {
      for(ii=0;(i=changed2dnum[ii])>=0;ii++) 
      {
	get_kernel_row(i,inactive2dnum,aicache);
	for(jj=0;(j=inactive2dnum[jj])>=0;jj++) 
        {
	  kernel_val=aicache[j];
	  lin[j]+=(((a[i]*kernel_val)-(a_old[i]*kernel_val))*(double)label[i]);
	}
      }
    }
  }
  (*maxdiff)=0;
  for(i=0;i<totdoc;i++) 
  {
    inactive_since[i]=deactnum-1;
    if(!inconsistent[i]) 
    {
      dist=(lin[i]-model.b)*(double)label[i];
      ex_c=learn_parm.svm_cost[i]-learn_parm.epsilon_a;
      if((a[i]>learn_parm.epsilon_a) && (dist > 1)) 
      {
	if((dist-1.0)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-1.0;
      }
      else if((a[i]<ex_c) && (dist < 1)) 
      {
	if((1.0-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=1.0-dist;
      }
      if((a[i]>(0+learn_parm.epsilon_a)) && (a[i]<ex_c)) 
      { 
	active[i]=1;                            /* not at bound */
      }
      else if((a[i]<=(0+learn_parm.epsilon_a)) && (dist < (1+learn_parm.epsilon_shrink))) 
      {
	active[i]=1;
      }
      else if((a[i]>=ex_c) && (dist > (1-learn_parm.epsilon_shrink))) 
      {
	active[i]=1;
      }
    }
  }
  for(i=0;i<totdoc;i++) 
  {
    (a_history[deactnum-1])[i]=a[i];
  }
  for(t=deactnum-2;(t>=0) && a_history[t];t--) 
  {
      delete a_history[t];
      a_history[t]=0;
  }
  delete changed;
  delete changed2dnum;
  delete inactive;
  delete inactive2dnum;
}

void svm::get_kernel_row(long docnum,long *active2dnum,CFLOAT *buffer)
{
  register long i,j,start;
  DOC *ex;

  ex=&(docs[docnum]);
  if(kernel_cache.index[docnum] != -1) 
  {
    /* is cached? */
    kernel_cache.lru[kernel_cache.index[docnum]] = kernel_cache.time; /* lru */
    start = kernel_cache.activenum * kernel_cache.index[docnum];
    for(i=0;(j=active2dnum[i])>=0;i++) 
    {
      if(kernel_cache.totdoc2active[j] >= 0) 
      {
	buffer[j] = kernel_cache.buffer[start+kernel_cache.totdoc2active[j]];
      }
      else 
      {
	buffer[j]=(CFLOAT)kernel(ex->words,docs[j].words);
      }
    }
  }
  else 
  {
    for(i=0;(j=active2dnum[i])>=0;i++) 
    {
      buffer[j]=(CFLOAT)kernel(ex->words,docs[j].words);
    }
  }
}

void svm::cache_multiple_kernel_rows(long *key,long varnum)
{
  register DOC *ex;
  register long j,k,l;
  register CFLOAT *cache;
  register long i;
  long     m;

  for(i=0;i<varnum;i++)
  {
    m = key[i];
    if(kernel_cache.index[m] == -1)
    {  
      /* not cached yet*/
      cache = kernel_cache_clean_and_malloc(m);
      if(cache) 
      {
        l  = kernel_cache.totdoc2active[m];
        ex = &(docs[m]);
        for(j=0;j<kernel_cache.activenum;j++) 
        {  
          /* fill cache */
	  k = kernel_cache.active2totdoc[j];
          if((kernel_cache.index[k] != -1) && (l != -1) && (k != m)) 
          {
	    cache[j] = kernel_cache.buffer[kernel_cache.activenum * kernel_cache.index[k]+l];
          }
          else 
          {
	    cache[j] = kernel(ex->words,docs[k].words);
          } 
        }
      }
      else 
      {
        perror("Error: Kernel cache full! => increase cache size");
      }
    }
  }
}

void svm::kernel_cache_shrink(long numshrink,long *after)
{
  /* which correspond to examples marked  */
  register long i,j,jj,from=0,to=0,scount;     /* 0 in after. */
  long *keep;

  keep = new long[totdoc];

  for(j=0;j<totdoc;j++) 
  {
    keep[j]=1;
  }
  scount=0;
  for(jj=0;(jj<kernel_cache.activenum) && (scount<numshrink);jj++) 
  {
    j=kernel_cache.active2totdoc[jj];
    if(!after[j]) 
    {
      scount++;
      keep[j]=0;
    }
  }

  for(i=0;i<kernel_cache.max_elems;i++) 
  {
    for(jj=0;jj<kernel_cache.activenum;jj++) 
    {
      j=kernel_cache.active2totdoc[jj];
      if(!keep[j]) 
      {
	from++;
      }
      else 
      {
	kernel_cache.buffer[to]=kernel_cache.buffer[from];
	to++;
	from++;
      }
    }
  }

  kernel_cache.activenum=0;
  for(j=0;j<totdoc;j++) 
  {
    if((keep[j]) && (kernel_cache.totdoc2active[j] != -1)) 
    {
      kernel_cache.active2totdoc[kernel_cache.activenum]=j;
      kernel_cache.totdoc2active[j]=kernel_cache.activenum;
      kernel_cache.activenum++;
    }
    else 
    {
      kernel_cache.totdoc2active[j]=-1;
    }
  }

  kernel_cache.max_elems=(long)(kernel_cache.buffsize / kernel_cache.activenum);
  if(kernel_cache.max_elems>totdoc) 
  {
    kernel_cache.max_elems=totdoc;
  }

  delete keep;
}

void svm::kernel_cache_init()
{
  long i;

  kernel_cache.index         = new long[totdoc];
  kernel_cache.occu          = new long[totdoc]; 
  kernel_cache.lru           = new long[totdoc]; 
  kernel_cache.invindex      = new long[totdoc];
  kernel_cache.active2totdoc = new long[totdoc];
  kernel_cache.totdoc2active = new long[totdoc];
  kernel_cache.buffer        = new CFLOAT[kernel_cache_size*1024*1024];
  kernel_cache.buffsize      = (long)(kernel_cache_size*1024*1024/sizeof(CFLOAT));
  kernel_cache.max_elems     = (long)(kernel_cache.buffsize/totdoc);
  if(kernel_cache.max_elems>totdoc) 
  {
    kernel_cache.max_elems=totdoc;
  }

  kernel_cache.elems=0;   /* initialize cache */
  for(i=0;i<totdoc;i++) 
  {
    kernel_cache.index[i]=-1;
  }
  for(i=0;i<kernel_cache.max_elems;i++) 
  {
    kernel_cache.occu[i]=0;
    kernel_cache.invindex[i]=-1;
  }

  kernel_cache.activenum=totdoc;;
  for(i=0;i<totdoc;i++) 
  {
      kernel_cache.active2totdoc[i]=i;
      kernel_cache.totdoc2active[i]=i;
  }

  kernel_cache.time=0;  
} 

void svm::kernel_cache_cleanup()
{
  delete kernel_cache.index;
  delete kernel_cache.occu;
  delete kernel_cache.lru;
  delete kernel_cache.invindex;
  delete kernel_cache.active2totdoc;
  delete kernel_cache.totdoc2active;
  delete kernel_cache.buffer;
}

long svm::kernel_cache_malloc()
{
  long i;

  if(kernel_cache.elems < kernel_cache.max_elems) 
  {
    for(i=0;i<kernel_cache.max_elems;i++) 
    {
      if(!kernel_cache.occu[i]) 
      {
	kernel_cache.occu[i]=1;
	kernel_cache.elems++;
	return(i);
      }
    }
  }
  return(-1);
}

long svm::kernel_cache_free_lru() /* remove least recently used cache */
{                                     
  register long k,least_elem=-1,least_time;

  least_time=kernel_cache.time+1;
  for(k=0;k<kernel_cache.max_elems;k++) 
  {
    if(kernel_cache.invindex[k] != -1) 
    {
      if(kernel_cache.lru[k]<least_time) 
      {
	least_time = kernel_cache.lru[k];
	least_elem = k;
      }
    }
  }
  if(least_elem != -1) 
  {
    kernel_cache.occu[least_elem] = 0;
    kernel_cache.elems--;
    kernel_cache.index[kernel_cache.invindex[least_elem]] = -1;
    kernel_cache.invindex[least_elem]                     = -1;
    return(1);
  }
  return(0);
}


CFLOAT *svm::kernel_cache_clean_and_malloc(long docnum)
{
   /* element is removed. */
  long result;
  if((result = kernel_cache_malloc()) == -1) 
  {
    if(kernel_cache_free_lru()) 
    {
      result = kernel_cache_malloc();
    }
  }
  kernel_cache.index[docnum] = result;
  if(result == -1) 
  {
    return(0);
  }
  kernel_cache.invindex[result] = docnum;
  kernel_cache.lru[kernel_cache.index[docnum]] = kernel_cache.time; /* lru */
  return((CFLOAT *)((long)kernel_cache.buffer + (kernel_cache.activenum * sizeof(CFLOAT) * kernel_cache.index[docnum])));
}

long svm::kernel_cache_touch(long docnum)
{
  if(kernel_cache.index[docnum] != -1)
  {
    kernel_cache.lru[kernel_cache.index[docnum]]=kernel_cache.time; /* lru */
    return(1);
  }
  return(0);
}
  
double svm::sprod_ss(FVAL *a,FVAL *b) /* Skalarprodukt */
{
    register FVAL sum=0;
    long i;
    for(i=0;i<totwords;i++) sum += a[i] * b[i];
    return((double)sum);
}

void svm::add_vector_ns(double *vec_n,FVAL *vec_s,double faktor)
{
  long i;
  for(i=0;i<totwords;i++) vec_n[i] += (faktor * vec_s[i]);
}

double svm::sprod_ns(double *vec_n,FVAL *vec_s)
{
  long i;
  double sum=0;
  for(i=0;i<totwords;i++) sum += (vec_n[i] * vec_s[i]);
  return(sum);
}

long svm::max(long a,long b)
{
  if(a>b) return(a);
  else    return(b);
}

double *svm::optimize_qp(double *epsilon_crit,long nx,double *threshold)
/* start the optimizer and return the optimal values */
/* The HIDEO optimizer does not necessarily fully solve the problem. */
/* Since it requires a strictly positive definite hessian, the solution */
/* is restricted to a linear independent subset in case the matrix is */
/* only semi-definite. */
{
  long i,j;
  int result,result2;

  if(!primal) 
  { 
    primal     = new double[nx];
    dual       = new double[(nx+1)*2]; 
    nonoptimal = new long[nx];
    buffer     = new double[(nx+1)*2* (nx+1)*2 + nx*nx+2 * (nx+1)*2 + 2*nx + 1 + 2*nx + nx];
  }

  result=optimize_hildreth_despo(qp.opt_n,qp.opt_m,opt_precision,(*epsilon_crit),maxiter,lindep_sensitivity,
				 qp.opt_g,qp.opt_g0,qp.opt_ce,qp.opt_ce0,
				 qp.opt_low,qp.opt_up,primal,qp.opt_xinit,
				 dual,nonoptimal,buffer);
  if(result == NAN_SOLUTION) 
  {
    lindep_sensitivity*=2;  /* throw out linear dependent examples more */
                            /* generously */
    if(learn_parm.svm_maxqpsize>2) 
    {
      learn_parm.svm_maxqpsize--;  /* decrease size of qp-subproblems */
    }
    while(result == NAN_SOLUTION) 
    {
      /* Shaking things up should help in this unlikely case. */
      for(i=0;i<qp.opt_n;i++) 
      {
	qp.opt_g[i*qp.opt_n+i]+=1;
      }
      result=optimize_hildreth_despo(qp.opt_n,qp.opt_m,opt_precision,(*epsilon_crit),maxiter,
				     lindep_sensitivity,qp.opt_g,qp.opt_g0,qp.opt_ce,
				     qp.opt_ce0,qp.opt_low,qp.opt_up,
				     primal,qp.opt_xinit,dual,nonoptimal,buffer);
    }
    result=NAN_SOLUTION;
    precision_violations++;
  }

  if(result == NO_PROGRESS_PRIMAL_OPTIMAL) count_nppo+=2;
  else if(count_nppo > 0)                  count_nppo--;

  if((result == ONLY_ONE_VARIABLE) || (count_nppo >= 7)) 
  {
      /* the problem could be, that all examples in the working set */
      /* are linear dependent. I have never seen this happen on my data. */
      /* But shaking things up should help in this unlikely case. */
      count_nppo=0;
      for(i=0;i<qp.opt_n;i++) 
      {
	for(j=0;j<qp.opt_n;j++) 
        {
	  qp.opt_g[i*qp.opt_n+j]=0;
	}
	qp.opt_g[i*qp.opt_n+i]=0.5/qp.opt_up[i];
      }
      result2=optimize_hildreth_despo(qp.opt_n,qp.opt_m,opt_precision,
			      (*epsilon_crit),maxiter,lindep_sensitivity,
			      qp.opt_g,qp.opt_g0,qp.opt_ce,qp.opt_ce0,
			      qp.opt_low,qp.opt_up,primal,qp.opt_xinit,dual,
			      nonoptimal,buffer);
  }

  if((result == NO_PROGRESS_DUAL_OPTIMAL) || (result == PROGRESS_DUAL_NOT_PRIMAL_OPTIMAL))  
  {
    if(learn_parm.svm_maxqpsize>2) 
    {
      learn_parm.svm_maxqpsize--;  /* decrease size of qp-subproblems */
    }
  }

  if(result == PROGRESS_MAXITER_NOT_PRIMAL_OPTIMAL) 
  {
    /* increase the precision */
    maxiter+=10;
  }

  if(result == NO_PROGRESS_MAXITER) 
  {
    /* in any case, increase the precision */
    maxiter+=50;
    precision_violations++;
  }

  if(precision_violations > 50) 
  {
    precision_violations=0;
    (*epsilon_crit)*=10.0; 
  }	  

  if(qp.opt_m>0) (*threshold)=dual[1]-dual[0];
  else           (*threshold)=0;

  if(isnan(*threshold)) (*threshold)=0;

  return(primal);
}

int svm::optimize_hildreth_despo(long n,long m,double precision,double epsilon_crit,long maxiter,double lindep_sensitivity,double *g,double *g0,double *ce,double *ce0,double *low,double *up,double *primal,double *init,double *dual,long *lin_dependent,double *buffer)
{
  long i,j,k,from,to,n_indep;
  double sum,bmin=0,bmax=0,epsilon_hideo,dist,model_b;
  double *d,*d0,*ig,*dual_old,*temp;       
  double *g0_new,*ce_new,*ce0_new,*low_new,*up_new;
  double add;
  int result;
  double obj_before,obj_after; 
  long b1,b2;

  g0_new=&(buffer[0]);    /* claim regions of buffer */
  d=&(buffer[n]);
  d0=&(buffer[n+(n+m)*2*(n+m)*2]);
  ce_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2]);
  ce0_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n]);
  ig=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m]);
  dual_old=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n]);
  low_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2]);
  up_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2+n]);
  temp=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2+n+n]);

  b1=-1;
  b2=-1;
  for(i=0;i<n;i++) {   /* get variables with steepest descent and */
    sum=g0[i];         /* opposite sign */
    for(j=0;j<n;j++) 
      sum+=init[j]*g[i*n+j];
    sum=sum-ce[i];
    if((b1==-1) || (sum<bmin)) {
      bmin=sum;
      b1=i;
    }
    if((b2==-1) || (sum>bmax)) {
      bmax=sum;
      b2=i;
    }
  }

  lcopy_matrix(g,n,d);
  add=0;
  if(m==1) {
    /* if we have a biased hyperplane, then adding a constant to the */
    /* hessian does not change the solution. So that is done for examples */
    /* with zero diagonal entry, since HIDEO cannot handle them. */
    for(i=0;i<n;i++) {
      if(d[i*n+i]==0) {
	for(j=0;j<n;j++) {
	  for(k=0;k<n;k++) {
	    d[j*n+k]+=1.0*ce[j]*ce[k];
	  }
	}
	add+=1.0;
	i=0;
      }
    }
  }

  lswitchrk_matrix(d,n,b1,b2); /* switch, so that variables are better mixed */
  linvert_matrix(d,n,ig,lindep_sensitivity,lin_dependent);
  lswitchrk_matrix(ig,n,b1,b2);
  i=lin_dependent[b1];  /* now switch back */
  lin_dependent[b1]=lin_dependent[b2];
  lin_dependent[b2]=i;
  lcopy_matrix(g,n,d);   /* restore d matrix */
  if(add>0)
    for(j=0;j<n;j++) {
      for(k=0;k<n;k++) {
	d[j*n+k]+=add*ce[j]*ce[k];
      }
    }

  for(i=0;i<n;i++) {  /* fix linear dependent vectors */
    g0_new[i]=g0[i]+add*ce0[0]*ce[i];
  }
  if(m>0) ce0_new[0]=-ce0[0];
  for(i=0;i<n;i++) {  /* fix linear dependent vectors */
    if(lin_dependent[i]) {
      for(j=0;j<n;j++) {
	if(!lin_dependent[j]) {
	  g0_new[j]+=init[i]*d[i*n+j];
	}
      }
      if(m>0) ce0_new[0]-=(init[i]*ce[i]);
    }
  }
  from=0;   /* remove linear dependent vectors */
  to=0;
  n_indep=0;
  for(i=0;i<n;i++) {
    if(!lin_dependent[i]) {
      g0_new[n_indep]=g0_new[i];
      ce_new[n_indep]=ce[i]; 
      low_new[n_indep]=low[i];
      up_new[n_indep]=up[i];
      primal[n_indep]=init[i];
      n_indep++;
    }
    for(j=0;j<n;j++) {
      if((!lin_dependent[i]) && (!lin_dependent[j])) {
        ig[to]=ig[from];
	to++;
      }
      from++;
    }
  }

  if(n_indep<=1) /* cannot optimize with only one variable */
    return((int)ONLY_ONE_VARIABLE);

  obj_before=calculate_qp_objective(n,g,g0,init);
  
  result=solve_dual(n_indep,m,precision,epsilon_crit,maxiter,g,g0_new,ce_new,
		    ce0_new,low_new,up_new,primal,d,d0,ig,dual,dual_old,temp);
  
  if(m>0) 
    model_b=dual[1]-dual[0];
  else
    model_b=0;
  
  j=n_indep;
  for(i=n-1;i>=0;i--) {
    if(!lin_dependent[i]) {
      j--;
      primal[i]=primal[j];
    }
    else if((m==0) && (g[i*n+i]==0)) {
      /* if we use a biased hyperplane, each example with a zero diagonal */
      /* entry must have an alpha at the upper bound. Doing this */
      /* is essential for the HIDEO optimizer, since it cannot handle zero */
      /* diagonal entries in the hessian for the unbiased hyperplane case. */
      primal[i]=up[i];  
    }
    else {
      primal[i]=init[i];  /* leave as is */
    }
    temp[i]=primal[i];
  }

  obj_after=calculate_qp_objective(n,g,g0,primal);

  if(isnan(obj_after)) return((int)NAN_SOLUTION);
  if(obj_after >= obj_before) 
  {
    if(result == MAXITER_EXCEEDED)  return((int)NO_PROGRESS_MAXITER);
    else if(result == DUAL_OPTIMAL) return((int)NO_PROGRESS_DUAL_OPTIMAL);
    else                            return((int)NO_PROGRESS_PRIMAL_OPTIMAL);
  }

  if(result != PRIMAL_OPTIMAL) 
  {
    /* Check the precision of the alphas. If results of current optimization */
    /* violate KT-Conditions, relax the epsilon on the bounds on alphas. */
    epsilon_hideo=EPSILON_HIDEO;
    for(i=0;i<n;i++) {
      if(!lin_dependent[i]) {
	dist=-model_b*ce[i]; 
	dist+=(g0[i]+1.0);
	for(j=0;j<i;j++) {
	  dist+=(primal[j]*g[j*n+i]);
	}
	for(j=i;j<n;j++) {
	  dist+=(primal[j]*g[i*n+j]);
	}
	if((primal[i]<(up[i]-epsilon_hideo)) && (dist < (1.0-epsilon_crit))) {
	  epsilon_hideo=(up[i]-primal[i])*2.0;
	}
	else if((primal[i]>(low[i]+epsilon_hideo)) && (dist > (1.0+epsilon_crit))) {
	  epsilon_hideo=(primal[i]-low[i])*2.0;
	}
	/* printf("HIDEO: a[%d]=%f, dist=%f, b=%f, epsilon_hideo=%f\n",i,
	   primal[i],dist,model_b,epsilon_hideo);  */
      }
    }
    for(i=0;i<n;i++) {  /* clip alphas to bounds */
      if(primal[i]<=(low[i]+epsilon_hideo)) {
	primal[i]=low[i];
      }
      else if(primal[i]>=(up[i]-epsilon_hideo)) {
	primal[i]=up[i];
      }
    }
    
    obj_after=calculate_qp_objective(n,g,g0,primal);
    
    if(obj_after >= obj_before) {
      for(i=0;i<n;i++) {  /* use unclipped values */
	primal[i]=temp[i];
      }
      if(result == MAXITER_EXCEEDED) 
	return((int)PROGRESS_MAXITER_NOT_PRIMAL_OPTIMAL);      
      else if(result == DUAL_OPTIMAL) 
	return((int)PROGRESS_DUAL_NOT_PRIMAL_OPTIMAL);      
    }
  }
  if(result == MAXITER_EXCEEDED) 
    return((int)PROGRESS_MAXITER_AND_PRIMAL_OPTIMAL);
  else if(result == DUAL_OPTIMAL) 
    return((int)PROGRESS_DUAL_AND_PRIMAL_OPTIMAL); 
  else if(result == PRIMAL_OPTIMAL) 
    return((int)PROGRESS_PRIMAL_OPTIMAL); 
  else
    return((int)PROGRESS);
}


int svm::solve_dual(long n,long m,double precision,double epsilon_crit,long maxiter,double *g,double *g0,double *ce,double *ce0,double *low,double *up,double *primal,double *d,double *d0,double *ig,double *dual,double *dual_old,double *temp)
     /* Solves the dual using the method of Hildreth and D'Espo. */
     /* Can only handle problems with zero or exactly one */
     /* equality constraints. */
{
  long i,j,k,iter;
  double sum,w,maxviol,viol,temp1,temp2;
  double model_b,dist;
  long retrain,maxfaktor,primal_optimal=0;
  double epsilon_a=1E-15;
  double obj_before; 

  if((m<0) || (m>1)) 
    perror("SOLVE DUAL: inappropriate number of eq-constrains!");

  obj_before=calculate_qp_objective(n,g,g0,primal);

  for(i=0;i<2*(n+m);i++) {
    dual[i]=0;
    dual_old[i]=0;
  }
  for(i=0;i<n;i++) {   
    for(j=0;j<n;j++) {   /* dual hessian for box constraints */
      d[i*2*(n+m)+j]=ig[i*n+j];
      d[(i+n)*2*(n+m)+j]=-ig[i*n+j];
      d[i*2*(n+m)+j+n]=-ig[i*n+j];
      d[(i+n)*2*(n+m)+j+n]=ig[i*n+j];
    }
    if(m>0) {
      sum=0;              /* dual hessian for eq constraints */
      for(j=0;j<n;j++) {
	sum+=(ce[j]*ig[i*n+j]);
      }
      d[i*2*(n+m)+2*n]=sum;
      d[i*2*(n+m)+2*n+1]=-sum;
      d[(n+i)*2*(n+m)+2*n]=-sum;
      d[(n+i)*2*(n+m)+2*n+1]=sum;
      d[(n+n)*2*(n+m)+i]=sum;
      d[(n+n+1)*2*(n+m)+i]=-sum;
      d[(n+n)*2*(n+m)+(n+i)]=-sum;
      d[(n+n+1)*2*(n+m)+(n+i)]=sum;
      
      sum=0;
      for(j=0;j<n;j++) {
	for(k=0;k<n;k++) {
	  sum+=(ce[k]*ce[j]*ig[j*n+k]);
	}
      }
      d[(n+n)*2*(n+m)+2*n]=sum;
      d[(n+n)*2*(n+m)+2*n+1]=-sum;
      d[(n+n+1)*2*(n+m)+2*n]=-sum;
      d[(n+n+1)*2*(n+m)+2*n+1]=sum;
    } 
  }

  for(i=0;i<n;i++) {   /* dual linear component for the box constraints */
    w=0;
    for(j=0;j<n;j++) {
      w+=(ig[i*n+j]*g0[j]); 
    }
    d0[i]=up[i]+w;
    d0[i+n]=-low[i]-w;
  }

  if(m>0) {  
    sum=0;             /* dual linear component for eq constraints */
    for(j=0;j<n;j++) {
      for(k=0;k<n;k++) {
	sum+=(ce[k]*ig[k*n+j]*g0[j]); 
      }
    }
    d0[2*n]=ce0[0]+sum;
    d0[2*n+1]=-ce0[0]-sum;
  }

  maxviol=999999;
  iter=0;
  retrain=1;
  maxfaktor=1;
  while((retrain) && (maxviol > 0) && (iter < (maxiter*maxfaktor))) {
    iter++;
    
    while((maxviol > precision) && (iter < (maxiter*maxfaktor))) {
      iter++;
      maxviol=0;
      for(i=0;i<2*(n+m);i++) {
	sum=d0[i];
	for(j=0;j<2*(n+m);j++) {
	  sum+=d[i*2*(n+m)+j]*dual_old[j];
	}
	sum-=d[i*2*(n+m)+i]*dual_old[i];
	dual[i]=-sum/d[i*2*(n+m)+i];
	if(dual[i]<0) dual[i]=0;
	
	viol=fabs(dual[i]-dual_old[i]);
	if(viol>maxviol) 
	  maxviol=viol;
	dual_old[i]=dual[i];
      }
      /*
      printf("%d) maxviol=%20f precision=%f\n",iter,maxviol,precision); 
      */
    }
  
    if(m>0) {
      for(i=0;i<n;i++) {
	temp[i]=dual[i]-dual[i+n]+ce[i]*(dual[n+n]-dual[n+n+1])+g0[i];
      }
    } 
    else {
      for(i=0;i<n;i++) {
	temp[i]=dual[i]-dual[i+n]+g0[i];
      }
    }
    for(i=0;i<n;i++) {
      primal[i]=0;             /* calc value of primal variables */
      for(j=0;j<n;j++) {
	primal[i]+=ig[i*n+j]*temp[j];
      }
      primal[i]*=-1.0;
      if(primal[i]<=(low[i]+epsilon_a)) {  /* clip to feasible region */
	primal[i]=low[i];
      }
      else if(primal[i]>=(up[i]-epsilon_a)) {
	primal[i]=up[i];
      }
    }

    if(m>0) 
      model_b=dual[n+n+1]-dual[n+n];
    else
      model_b=0;
    retrain=0;
    primal_optimal=1;
    for(i=0;i<n;i++) {     /* check primal KT-Conditions */
      dist=-model_b*ce[i]; 
      dist+=(g0[i]+1.0);
      for(j=0;j<i;j++) {
	dist+=(primal[j]*g[j*n+i]);
      }
      for(j=i;j<n;j++) {
	dist+=(primal[j]*g[i*n+j]);
      }
      /* printf("HIDEOtemp: a[%d]=%f, dist=%f, b=%f\n",i,primal[i],dist,model_b); */
      if((primal[i]<(up[i]-epsilon_a)) && (dist < (1.0-epsilon_crit))) {
	retrain=1;
	primal_optimal=0;
      }
      else if((primal[i]>(low[i]+epsilon_a)) && (dist > (1.0+epsilon_crit))) {
	retrain=1;
	primal_optimal=0;
      }
    }
    if(retrain) 
      precision/=10;
    if(obj_before <= calculate_qp_objective(n,g,g0,primal)) { 
      maxfaktor=10;
      if(!retrain) {
	retrain=1;
	precision/=10;
      }
    }
    else {
      maxfaktor=1;
    }
  }

  if(m>0) {
    temp1=dual[n+n+1];   /* copy the dual variables for the eq */
    temp2=dual[n+n];     /* constraints to a handier location */
    for(i=n+n+1;i>=2;i--) {
      dual[i]=dual[i-2];
    }
    dual[0]=temp2;
    dual[1]=temp1;
  }

  if(primal_optimal) {
    return((int)PRIMAL_OPTIMAL);
  }
  else if(maxviol == 0.0) {
    return((int)DUAL_OPTIMAL);
  }
  else {
    return((int)MAXITER_EXCEEDED);
  }
}

void svm::linvert_matrix(double *matrix,long depth,double *inverse,double lindep_sensitivity,long *lin_dependent)
{
  long i,j,k;
  double factor;

  for(i=0;i<depth;i++) 
  {
    lin_dependent[i]=0;
    for(j=0;j<depth;j++) 
    {
      if(i==j) inverse[i*depth+j]=1.0;
      else inverse[i*depth+j]=0.0;
    }
  }
  for(i=0;i<depth;i++) 
  {
    if(fabs(matrix[i*depth+i])<lindep_sensitivity) 
    {
      lin_dependent[i]=1;
    }
    else 
    {
      for(j=i+1;j<depth;j++) 
      {
	factor=matrix[j*depth+i]/matrix[i*depth+i];
	for(k=i;k<depth;k++) 
        {
	  matrix[j*depth+k]-=(factor*matrix[i*depth+k]);
	}
	for(k=0;k<depth;k++) 
        {
	  inverse[j*depth+k]-=(factor*inverse[i*depth+k]);
	}
      }
    }
  }
  for(i=depth-1;i>=0;i--) 
  {
    if(!lin_dependent[i]) 
    {
      factor=1/matrix[i*depth+i];
      for(k=0;k<depth;k++) 
      {
	inverse[i*depth+k]*=factor;
      }
      matrix[i*depth+i]=1;
      for(j=i-1;j>=0;j--) 
      {
	factor=matrix[j*depth+i];
	matrix[j*depth+i]=0;
	for(k=0;k<depth;k++) 
        {
	  inverse[j*depth+k]-=(factor*inverse[i*depth+k]);
	}
      }
    }
  }
}

void svm::lcopy_matrix(double *matrix,long depth,double *matrix2)
{
  long i;
  
  for(i=0;i<(depth)*(depth);i++) 
  {
    matrix2[i]=matrix[i];
  }
}

void svm::lswitchrk_matrix(double *matrix,long depth,long rk1,long rk2)
{
  long i;
  double temp;

  for(i=0;i<depth;i++) 
  {
    temp=matrix[rk1*depth+i];
    matrix[rk1*depth+i]=matrix[rk2*depth+i];
    matrix[rk2*depth+i]=temp;
  }
  for(i=0;i<depth;i++) 
  {
    temp=matrix[i*depth+rk1];
    matrix[i*depth+rk1]=matrix[i*depth+rk2];
    matrix[i*depth+rk2]=temp;
  }
}

double svm::calculate_qp_objective(long opt_n,double *opt_g,double *opt_g0,double *alpha)
{
  double obj;
  long i,j;
  obj=0;  /* calculate objective  */
  for(i=0;i<opt_n;i++) 
  {
    obj+=(opt_g0[i]*alpha[i]);
    obj+=(0.5*alpha[i]*alpha[i]*opt_g[i*opt_n+i]);
    for(j=0;j<i;j++) 
    {
      obj+=(alpha[j]*alpha[i]*opt_g[j*opt_n+i]);
    }
  }
  return(obj);
}

