import numpy as np

class SGDMF():
    def __init__(self, 
                 ratings,
                 n_factors=40,
                 item_reg=0.1, 
                 user_reg=0.1,
                 item_bias_reg=0.1,
                 user_bias_reg=0.1,
                 pct_train=1.0,
                ):
        """
        Train an SGD matrix factorization model to predict empty 
        entries in a matrix. 
        """
        
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.sample_row, self.sample_col = self.ratings.nonzero()
        self.n_samples = len(self.sample_row)
        self.pct_train = pct_train

    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            # make prediction and calculate errors
            prediction = self.predict_rating(u, i)
            error = (self.ratings[u,i] - prediction) 
            
            # Update biases and latent factors
            # according to mathematical derivation
            self.user_bias[u] += self.learning_rate * \
                                (error - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * \
                                (error - self.item_bias_reg * self.item_bias[i])
            self.user_vecs[u, :] += self.learning_rate * \
                                   (error * self.item_vecs[i, :] - \
                                     self.user_reg * self.user_vecs[u,:])
            self.item_vecs[i, :] += self.learning_rate * \
                                   (error * self.user_vecs[u, :] - \
                                    self.item_reg * self.item_vecs[i,:])
    
    def train(self, n_iter=10, learning_rate=0.1, pct_train = 0.8):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors        
        self.user_vecs = np.random.random(size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.random(size=(self.n_items, self.n_factors))

        self.learning_rate = learning_rate
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
        
        epoch = 1
        while epoch <= n_iter:
            self.training_indices = np.arange(self.n_samples)
            # each epoch using a pct train
            np.random.shuffle(self.training_indices)
            self.training_indices = np.random.choice(self.training_indices,
                                                     size = int(self.n_samples * pct_train),
                                                     replace = False)
            self.sgd()
            print("Finished epoch", epoch, "of", n_iter)
            epoch += 1

    
    def predict_rating(self, u, i):
        prediction = self.global_bias \
                    + self.user_bias[u] \
                    + self.item_bias[i] \
                    + self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        return prediction

    
    # def train_test_split(self, ratings, pct):
    #     test_set = np.zeros(ratings.shape)
    #     train_set = ratings.copy()
    #     for user in range(ratings.shape[0]):
    #         user_rating_idx = ratings[user, :].nonzero()[0]
    #         test_ratings = np.random.choice(user_rating_idx,
    #                                         size=int(len(user_rating_idx)*pct),
    #                                         replace=False)
    #         train_set[user, test_ratings] = 0.
    #         test_set[user, test_ratings] = ratings[user, test_ratings]
            
    #     # Test and training are truly disjoint
    #     assert(np.all((train_set * test_set) == 0)) 
    #     return train_set, test_set