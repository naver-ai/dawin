import numpy as np
from scipy.special import betaln, logsumexp
from sklearn.cluster import KMeans

class BetaMixtureModel:
    """
    Beta Mixture Model (Multivariate version).
    Each dimension is modeled independently by a Beta distribution.
    """

    def __init__(self, n_mixtures=3, random_seed=1):
        self.n_mixtures = n_mixtures
        self.random_seed = random_seed
        self.convergence = False

    def _init_clusters(self, data_matrix, init_round):
        """
        Initialize the mixture responsibilities (assignments) via k-means or uniformly random
        """
        if self.method == "kmeans":
            km = KMeans(
                n_clusters=self.n_mixtures,
                n_init=1,
                random_state=self.random_seed + init_round
            ).fit(data_matrix)
            resp_matrix = np.zeros((self.n_observations, self.n_mixtures))
            resp_matrix[np.arange(self.n_observations), km.labels_] = 1
        else:
            np.random.seed(self.random_seed + init_round)
            resp_matrix = np.random.rand(self.n_observations, self.n_mixtures)
            resp_matrix /= resp_matrix.sum(axis=1, keepdims=True)
        
        # Numerical stability
        resp_matrix += 10 * np.finfo(resp_matrix.dtype).eps

        # Initialize beta parameters (alpha/beta for each dimension)
        self.beta_params_ = np.zeros((self.n_mixtures, self.n_components * 2))
        self._M_step(data_matrix, np.log(resp_matrix))


    def _calc_log_weights(self):
        """
        Return log of current mixture weights.
        """
        return np.log(self.mix_weights_)

    def _calc_mixture_log_probs(self, data_matrix, mixture_idx):
        """
        Compute log-prob for a single mixture (used if parallelized).
        """
        alpha_vec = self.beta_params_[mixture_idx, :self.n_components]
        beta_vec = self.beta_params_[mixture_idx, self.n_components:]
        beta_func_log = betaln(alpha_vec, beta_vec)
        return (
            (alpha_vec - 1) * np.log(data_matrix)
            + (beta_vec - 1) * np.log(1 - data_matrix)
            - beta_func_log
        ).sum(axis=1)

    def _calc_log_probs_all_mixtures(self, data_matrix):
        """
        Return log-prob for each observation under each mixture (unnormalized).
        """
        log_prob = np.empty((self.n_observations, self.n_mixtures))
        for mix in range(self.n_mixtures):
            alpha_vec = self.beta_params_[mix, :self.n_components]
            beta_vec = self.beta_params_[mix, self.n_components:]
            bfn = betaln(alpha_vec, beta_vec)
            log_prob[:, mix] = (
                (alpha_vec - 1) * np.log(data_matrix)
                + (beta_vec - 1) * np.log(1 - data_matrix)
                - bfn
            ).sum(axis=1)
        return log_prob

    def _calc_weighted_log_probs(self, data_matrix):
        """
        Return the sum of log-probabilities and log-weights.
        """
        return self._calc_log_probs_all_mixtures(data_matrix) + self._calc_log_weights()

    def _calc_log_resp_and_norm(self, data_matrix):
        """
        Return (log_prob_norm, log_resp) for the E-step.
        """
        weighted_lp = self._calc_weighted_log_probs(data_matrix)
        lp_norm = logsumexp(weighted_lp, axis=1)
        with np.errstate(under="ignore"):
            log_resp = weighted_lp - lp_norm[:, None]
        return lp_norm, log_resp

    def _E_step(self, data_matrix):
        """
        E-step: compute average log_prob_norm and log_resp.
        """
        lp_norm, log_resp = self._calc_log_resp_and_norm(data_matrix)
        return np.mean(lp_norm), log_resp

    def _compute_responsibilities(self, log_resp):
        """
        Exponentiate log_resp and sum across observations.
        """
        resp_matrix = np.exp(log_resp)
        cluster_counts = resp_matrix.sum(axis=0) + 10 * np.finfo(resp_matrix.dtype).eps
        return resp_matrix, cluster_counts

    def _update_mixture_weights(self, cluster_counts):
        """
        Update mixture weights from mixture counts.
        """
        self.mix_weights_ = cluster_counts / cluster_counts.sum()

    def _M_step(self, data_matrix, log_resp):
        """
        M-step: update weights and Beta distribution parameters via moment matching.
        """
        resp_matrix, cluster_counts = self._compute_responsibilities(log_resp)
        self._update_mixture_weights(cluster_counts)

        w_sums = resp_matrix.T @ data_matrix
        w_sums_sq = resp_matrix.T @ (data_matrix ** 2)

        for m_idx in range(self.n_mixtures):
            sum_vals = w_sums[m_idx]
            sum_sq_vals = w_sums_sq[m_idx]
            mean_val = sum_vals / cluster_counts[m_idx]
            var_val = sum_sq_vals / cluster_counts[m_idx] - mean_val ** 2

            # Clip variance
            variance_cap = mean_val * (1 - mean_val) / 4
            var_val = np.minimum(var_val, variance_cap)
            var_val += 10 * np.finfo(var_val.dtype).eps

            # Compute factor
            scaling_factor = (mean_val * (1 - mean_val)) / (var_val + 1e-10) - 1
            self.beta_params_[m_idx, :self.n_components] = scaling_factor * mean_val
            self.beta_params_[m_idx, self.n_components:] = scaling_factor * (1 - mean_val)

    def fit(self, data_matrix, num_init=3, method="kmeans", max_iter=1000, tol=1e-4):
        """
        Fit BetaMixtureModel to the data using EM, possibly with multiple initializations.
        """
        self.n_observations, self.n_components = data_matrix.shape
        self.convergence = False
        self.method = method
        best_lower_bound = -np.inf
        optimal_params = None

        for init_round in range(num_init):
            print(f"{init_round + 1}-th BMM initialization")
            self._init_clusters(data_matrix, init_round)
            ll_bound = -np.inf

            for _ in range(max_iter):
                prev_bound = ll_bound
                lp_norm, log_resp = self._E_step(data_matrix)
                self._M_step(data_matrix, log_resp)
                ll_bound = lp_norm
                delta_bound = ll_bound - prev_bound

                if abs(delta_bound) < tol:
                    self.convergence = True
                    break

            if ll_bound > best_lower_bound:
                best_lower_bound = ll_bound
                # Update final weights
                _, cluster_counts = self._compute_responsibilities(log_resp)
                self._update_mixture_weights(cluster_counts)
                optimal_params = (self.mix_weights_.copy(), self.beta_params_.copy())

        self.mix_weights_, self.beta_params_ = optimal_params
        self.max_lower_bound = best_lower_bound
        return self

    def predict_proba(self, data_matrix):
        """
        Return the per-mixture membership probabilities for each sample.
        """
        _, log_resp = self._calc_log_resp_and_norm(data_matrix)
        return np.exp(log_resp)

    def predict(self, data_matrix):
        """
        Return the most probable mixture index for each sample.
        """
        return np.argmax(self.predict_proba(data_matrix), axis=1)