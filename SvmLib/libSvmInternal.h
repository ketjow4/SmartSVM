#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 323
#include <algorithm>
#include <vector>
#include <utility>
#include <string>

#include "Feature.h"

/*#ifdef __cplusplus
extern "C" {
#endif*/

    extern int libsvm_version;

    struct svm_node
    {
        int index;
        double value;
    };

    struct svm_problem
    {
        int l;
        double *y;
        struct svm_node **x;
    };

    enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
    enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED, RBF_CUSTOM, RBF_SUM, RBF_SUM_DIV2,
    	RBF_DIV,  RBF_MAX, RBF_MIN, RBF_SUM_2_KERNELS, RBF_LINEAR, RBF_POLY_GLOBAL,
    	RBF_LINEAR_MAX,
		RBF_LINEAR_MIN,
		RBF_LINEAR_SUM_2_KERNELS, RBF_LINEAR_SINGLE,
	}; /* kernel_type */

    struct svm_parameter
    {
        int svm_type;
        int kernel_type;
        int degree;	/* for poly */
        double gamma;	/* for poly/rbf/sigmoid */
        double coef0;	/* for poly/sigmoid */
		double t; /*used for mixed kernel based on https://iopscience.iop.org/article/10.1088/1742-6596/1187/4/042063/pdf*/

                        /* these are for training only */
        double cache_size; /* in MB */
        double eps;	/* stopping criteria */
        double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
        int nr_weight;		/* for C_SVC */
        int *weight_label;	/* for C_SVC */
        double* weight;		/* for C_SVC */
        double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
        double p;	/* for EPSILON_SVR */
        int shrinking;	/* use the shrinking heuristics */
        int probability; /* do probability estimates */
		double m_optimalProbabilityThreshold;

		bool m_optimalThresholdSet;
		double m_certaintyNegative;
		double m_certaintyPositive;
		double m_certaintyNegativeNormalized;
		double m_certaintyPositiveNormalized;

		double m_certaintyNegativeClassOnly;
		double m_certaintyPositiveClassOnly;
		double m_certaintyNegativeNormalizedClassOnly;
		double m_certaintyPositiveNormalizedClassOnly;
    	
		bool reached_max_iter;
		bool trainAlpha;  //when this is false alpha's are set to 1 or -1 (depending on class) and not optimized during training otherwise regular svm training is performed

        std::vector<double>* gammas; //used only during training
		std::vector<double>* gammas_after_training; //used for predictions after finished training
		std::vector<svmComponents::Feature>* features;
    };

    //
    // svm_model
    //
    struct svm_model
    {
        struct svm_parameter param;	/* parameter */
        int nr_class;		/* number of classes, = 2 in regression/one class svm */
        int l;			/* total #SV */
        struct svm_node **SV;		/* SVs (SV[l]) */
        double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
        double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
        double *probA;		/* pariwise probability information */
        double *probB;
        int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

                                /* for classification only */

        int *label;		/* label of each class (label[k]) */
        int *nSV;		/* number of SVs for each class (nSV[k]) */
                        /* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
                        /* XXX */
        int free_sv;		/* 1 if svm_model is created by svm_load_model*/
                            /* 0 if svm_model is created by svm_train */
		std::string* groupStrategyName;
    };

    struct svm_model *svm_train(const struct svm_problem *prob, struct svm_parameter *param);
    void svm_cross_validation(const struct svm_problem *prob, struct svm_parameter *param, int nr_fold, double *target);

    int svm_save_model(const char *model_file_name, const struct svm_model *model);
	std::string svm_save_model_to_string(const svm_model* model);

    struct svm_model *svm_load_model(const char *model_file_name);
	svm_model* svm_load_model_from_string(std::string model_text);

    int svm_get_svm_type(const struct svm_model *model);
    int svm_get_nr_class(const struct svm_model *model);
    void svm_get_labels(const struct svm_model *model, int *label);
    void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
    int svm_get_nr_sv(const struct svm_model *model);
    double svm_get_svr_probability(const struct svm_model *model);

    double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);

	std::tuple<double, double, int> svm_predict_values_pos_neg(const svm_model* model, const svm_node* x);
	std::pair<double, double> svm_predict_values_with_closest_distance(const svm_model* model, const svm_node* x, double* dec_values);

    double svm_predict(const struct svm_model *model, const struct svm_node *x);
    double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

    void svm_free_model_content(struct svm_model *model_ptr);
    void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
    void svm_destroy_param(struct svm_parameter *param);

    const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
    int svm_check_probability_model(const struct svm_model *model);

    void svm_set_print_string_function(void(*print_func)(const char *));


	

	//Extracted from cpp for check_sv function 

	typedef float Qfloat;

    template <class T>
    static inline void swap(T& x, T& y)
    {
	    T t = x;
	    x = y;
	    y = t;
    }

	static inline double powi(double base, int times)
	{
		double tmp = base, ret = 1.0;

		for (int t = times; t > 0; t /= 2)
		{
			if (t % 2 == 1) ret *= tmp;
			tmp = tmp * tmp;
		}
		return ret;
	}
//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
	class QMatrix {
	public:
		virtual Qfloat *get_Q(int column, int len) const = 0;
		virtual double *get_QD() const = 0;
		virtual void swap_index(int i, int j) const = 0;
		virtual ~QMatrix() {}
	};

	class Kernel : public QMatrix {
	public:
		Kernel(int l, svm_node * const * x, const svm_parameter& param);
		virtual ~Kernel();

		static double k_function(const svm_node *x, const svm_node *y,
			const svm_parameter& param, int y_index);

		//original version
		//virtual Qfloat *get_Q(int column, int len) const = 0;
		//virtual double *get_QD() const = 0;
		virtual Qfloat* get_Q(int /*column*/, int /*len*/) const { throw std::exception("Not implemented"); };
		virtual double* get_QD() const { throw std::exception("Not implemented");};
		virtual void swap_index(int i, int j) const	// no so const...
		{
			swap(x[i], x[j]);
			if (x_square) swap(x_square[i], x_square[j]);
		}
		std::vector<double> gammas;


		double kernerl_rbf_and_linear(int i, int j) const
		{
			if (gammas[i] < 0 && gammas[j] < 0)
			{
				auto line = dot(x[i], x[j]);
				return line;
			}
			else if (gammas[i] < 0)
			{
				auto g = exp(-(gammas[j]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
				return g;
			}
			else if (gammas[j] < 0)
			{
				auto g = exp(-(gammas[i]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
				return g;
			}
			//sum of two gammas
			switch (kernel_type)
			{
			case RBF_LINEAR:
				return  exp(-(gammas[i] + gammas[j]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
			case RBF_LINEAR_SINGLE:
				return  exp(-(gammas[i]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
			case RBF_LINEAR_MAX:
				return exp(-std::max(gammas[i], gammas[j]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
			case RBF_LINEAR_MIN:
				return exp(-std::min(gammas[i], gammas[j]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
			case RBF_LINEAR_SUM_2_KERNELS:
				return exp(-gammas[i] * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])))
					+ exp(-gammas[j] * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
			default:
				throw std::exception("Wrong kernel rbf + linear, libsvmInternal.h line 207");
			}			
			
		}


	protected:
		double (Kernel::* kernel_function)(int i, int j) const;
		

	private:
		const svm_node **x;
		double *x_square;

		// svm_parameter
		const int kernel_type;
		const int degree;
		const double gamma;
		const double coef0;
		
		const double t;

		static double dot(const svm_node *px, const svm_node *py);
		double kernel_linear(int i, int j) const
		{
			return dot(x[i], x[j]);
		}
		double kernel_poly(int i, int j) const
		{
			return powi(gamma*dot(x[i], x[j]) + coef0, degree);
		}
		double kernel_rbf(int i, int j) const
		{
			return exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		}
		double kernel_sigmoid(int i, int j) const
		{
			return tanh(gamma*dot(x[i], x[j]) + coef0);
		}
		double kernel_precomputed(int i, int j) const
		{
			return x[i][(int)(x[j][0].value)].value;
		}

		double kernerl_rbf_custom(int i, int j) const
		{
			//RBF_CUSTOM
			//gamma is taken from first i-th vector
			return exp(-(gammas[i]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		}

		double kernerl_rbf_sum(int i, int j) const
		{
			return exp(-(gammas[i] + gammas[j]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		}

		//@wdudzik gives the same result as kernerl_rbf_sum as this division is uniform scalling applied and do not change the results of calculation
		double kernerl_rbf_sum_div2(int i, int j) const
		{
			return exp(-((gammas[i] + gammas[j]) / 2) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		}

		//poor results of this function, do not use in practice
		double kernerl_rbf_div(int i, int j) const
		{

			return exp(-(gammas[i] / gammas[j]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		}

		double kernerl_rbf_min(int i, int j) const
		{

			return exp(-std::min(gammas[i], gammas[j]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		}

		double kernerl_rbf_max(int i, int j) const
		{

			return exp(-std::max(gammas[i], gammas[j]) * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		}

		double kernerl_rbf_sum_2_kernels(int i, int j) const
		{
			return exp(-gammas[i] * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])))
			+ exp(-gammas[j] * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		}


		//RBF_POLY_GLOBAL -- based on https://iopscience.iop.org/article/10.1088/1742-6596/1187/4/042063/pdf
		double kernerl_rbf_poly_global(int i, int j) const
		{
			return (1 - t) *powi( dot(x[i], x[j]) + coef0, degree)
			+ t * exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		}
	};


//#ifdef __cplusplus
//}
//#endif

#endif /* _LIBSVM_H */
