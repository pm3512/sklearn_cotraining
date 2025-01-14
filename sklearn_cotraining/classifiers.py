import numpy as np
import random
import copy

from sklearn.base import BaseEstimator

from utils import compute_conditionals, compute_conditionals_one_view, compute_conditionals_view_to_view, compute_posteriors, compute_prob_one_view, supports_proba, compute_posteriors_one_view
class CoTrainingClassifier(object):
	"""
	Parameters:
	clf - The classifier that will be used in the cotraining algorithm on the X1 feature set
		(Note a copy of clf will be used on the X2 feature set if clf2 is not specified).

	clf2 - (Optional) A different classifier type can be specified to be used on the X2 feature set
		 if desired.

	p - (Optional) The number of positive examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	n - (Optional) The number of negative examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	k - (Optional) The number of iterations
		The default is 30 (from paper)

	u - (Optional) The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 (from paper)
	"""

	def __init__(self, clf, clf2=None, p=-1, n=-1, k=40, u=75, num_classes=2):
		self.clf1_ = clf

		#we will just use a copy of clf (the same kind of classifier) if clf2 is not specified
		if clf2 == None:
			self.clf2_ = copy.copy(clf)
		else:
			self.clf2_ = clf2

		#if they only specify one of n or p, through an exception
		if (p == -1 and n != -1) or (p != -1 and n == -1):
			raise ValueError('Current implementation supports either both p and n being specified, or neither')

		self.p_ = p
		self.n_ = n
		self.k_ = k
		self.u_ = u
		self.num_classes_ = num_classes

		random.seed()


	def fit(self, X1, X2, y, y1_true, y2_true, y_true):
		"""
		Description:
		fits the classifiers on the partially labeled data, y.

		Parameters:
		X1 - array-like (n_samples, n_features_1): first set of features for samples
		X2 - array-like (n_samples, n_features_2): second set of features for samples
		y - array-like (n_samples): labels for samples, -1 indicates unlabeled

		"""

		#we need y to be a numpy array so we can do more complex slicing
		y = np.asarray(y)

		#set the n and p parameters if we need to
		if self.p_ == -1 and self.n_ == -1:
			num_pos = sum(1 for y_i in y if y_i == 1)
			num_neg = sum(1 for y_i in y if y_i == 0)

			n_p_ratio = num_neg / float(num_pos)

			if n_p_ratio > 1:
				self.p_ = 1
				self.n_ = round(self.p_*n_p_ratio)

			else:
				self.n_ = 1
				self.p_ = round(self.n_/n_p_ratio)

		assert(self.p_ > 0 and self.n_ > 0 and self.k_ > 0 and self.u_ > 0)

		#the set of unlabeled samples
		U = [i for i, y_i in enumerate(y) if y_i == -1]

		#we randomize here, and then just take from the back so we don't have to sample every time
		random.Random(1).shuffle(U)

		#this is U' in paper
		U_ = U[-min(len(U), self.u_):]

		#the samples that are initially labeled
		L = [i for i, y_i in enumerate(y) if y_i != -1]

		#remove the samples in U_ from U
		U = U[:-len(U_)]


		it = 0 #number of cotraining iterations we've done so far

		#loop until we have assigned labels to everything in U or we hit our iteration break condition
		while it != self.k_ and U:
			it += 1

			self.clf1_.fit(X1[L], y[L])
			self.clf2_.fit(X2[L], y[L])

			y1_prob = self.clf1_.predict_proba(X1[U_])
			y2_prob = self.clf2_.predict_proba(X2[U_])

			n1, p1 = [], []
			n2, p2 = [], []

			if self.num_classes_ == 2:
				for i in (y1_prob[:,0].argsort())[-self.n_:]:
					#if y1_prob[i,0] > 0.5:
					n2.append(i)
				for i in (y1_prob[:,1].argsort())[-self.p_:]:
					#if y1_prob[i,1] > 0.5:
					p2.append(i)

				for i in (y2_prob[:,0].argsort())[-self.n_:]:
					#if y2_prob[i,0] > 0.5:
					n1.append(i)
				for i in (y2_prob[:,1].argsort())[-self.p_:]:
					#if y2_prob[i,1] > 0.5:
					p1.append(i)
				p = set(p1 + p2)
				n = set(n1 + n2)

				#label the samples and remove thes newly added samples from U_
				y[[U_[x] for x in p]] = 1
				y[[U_[x] for x in n]] = 0
				y_pred = np.concatenate((y[[U_[x] for x in p]], y[[U_[x] for x in n]]))
				y_true_sub = np.concatenate((y_true[[U_[x] for x in p]], y_true[[U_[x] for x in n]]))
				print('cotrain iteration %d: y1+y2->y_true acc %f' % (it, (y_pred == y_true_sub).mean()))
				y1_pred = np.concatenate([y[[U_[x] for x in p1]], y[[U_[x] for x in n1]]])
				y1_true_sub = np.concatenate((y1_true[[U_[x] for x in p1]], y1_true[[U_[x] for x in n1]]))
				y2_pred = np.concatenate([y[[U_[x] for x in p2]], y[[U_[x] for x in n2]]])
				y2_true_sub = np.concatenate((y2_true[[U_[x] for x in p2]], y2_true[[U_[x] for x in n2]]))
				print('y1->y2 acc %f, y2->y1 acc %f' % ((y2_pred == y2_true_sub).mean(), (y1_pred == y1_true_sub).mean()))

				L.extend([U_[x] for x in p])
				L.extend([U_[x] for x in n])

				U_ = [elem for i, elem in enumerate(U_) if not (i in p or i in n)]

				#add new elements to U_
				num_to_add = len(p) + len(n)
				num_to_add = min(num_to_add, len(U))
				U_.extend(U[-num_to_add:])
				U = U[:-num_to_add]

			else:
				new_labels = {}
				for i in range(self.num_classes_):
					new_labels[i] = set()
					for j in (y1_prob[:,i].argsort())[-self.n_:]:
						if y1_prob[j,i] > 0.5:
							new_labels[i].add(j)
					for j in (y2_prob[:,i].argsort())[-self.p_:]:
						if y2_prob[j,i] > 0.5:
							new_labels[i].add(j)
				
				for i in range(self.num_classes_):
					y[[U_[x] for x in new_labels[i]]] = i
					L.extend([U_[x] for x in new_labels[i]])
				new_labels_all = set()
				for i in range(self.num_classes_):
					new_labels_all.update(new_labels[i])
				U_ = [elem for j, elem in enumerate(U_) if not (j in new_labels_all)]				

				num_to_add = sum(len(new_labels_class) for new_labels_class in new_labels.values())
				num_to_add = min(num_to_add, len(U))
				U_.extend(U[-num_to_add:])
				U = U[:-num_to_add]


		#let's fit our final model
		self.clf1_.fit(X1[L], y[L])
		self.clf2_.fit(X2[L], y[L])


	def supports_proba(self, clf):
		"""Checks if a given classifier supports the 'predict_proba' method, given a single vector x"""
		return hasattr(clf, 'predict_proba')

	def predict(self, X1, X2):
		"""
		Predict the classes of the samples represented by the features in X1 and X2.

		Parameters:
		X1 - array-like (n_samples, n_features1)
		X2 - array-like (n_samples, n_features2)


		Output:
		y - array-like (n_samples)
			These are the predicted classes of each of the samples.  If the two classifiers, don't agree, we try
			to use predict_proba and take the classifier with the highest confidence and if predict_proba is not implemented, then we randomly
			assign either 0 or 1.  We hope to improve this in future releases.

		"""

		y1 = self.clf1_.predict(X1)
		y2 = self.clf2_.predict(X2)

		proba_supported = supports_proba(self.clf1_) and supports_proba(self.clf2_)

		#fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree
		y_pred = np.asarray([-1] * X1.shape[0])

		for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
			if y1_i == y2_i:
				y_pred[i] = y1_i
			elif proba_supported:
				y1_probs = np.array(self.clf1_.predict_proba([X1[i]])[0])
				y2_probs = np.array(self.clf2_.predict_proba([X2[i]])[0])
				sum_probs = y1_probs + y2_probs
				pred = sum_probs.argmax()
				y_pred[i] = pred

			else:
				#the classifiers disagree and don't support probability, so we guess
				y_pred[i] = random.randint(0, 1)


		#check that we did everything right
		assert not (-1 in y_pred)

		return y_pred

class SeparateViewsClassifier(object):
	def __init__(self, clf: BaseEstimator, clf2: BaseEstimator=None):
		self.clf1_ = clf

		#we will just use a copy of clf (the same kind of classifier) if clf2 is not specified
		if clf2 == None:
			self.clf2_ = copy.copy(clf)
		else:
			self.clf2_ = clf2
	
	def fit(self, X1: np.ndarray, X2: np.ndarray, y: np.ndarray):
		y = np.asarray(y)
		L = (y != -1)
		self.clf1_.fit(X1[L], y[L])
		self.clf2_.fit(X2[L], y[L])
	
	def predict_proba(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
		if not supports_proba(self.clf1_) or not supports_proba(self.clf2_):
			raise Exception("The classifiers don't support predict_proba")
		probas_1 = self.clf1_.predict_proba(X1)
		probas_2 = self.clf2_.predict_proba(X2)
		probas = (probas_1 + probas_2) / 2
		return probas
	
	def predict(self, X1, X2) -> np.ndarray:
		probas = self.predict_proba(X1, X2)
		return np.argmax(probas, axis=1)

class DistributionAwarePred(object):
	"""
	Parameters:
	clf - The classifier that will be used in the cotraining algorithm on the X1 feature set
		(Note a copy of clf will be used on the X2 feature set if clf2 is not specified).

	prob_tensor: 3D tensor of shape (n_classes, n_classes, n_classes), which gives the
	distribution of sample classes

	clf2 - (Optional) A different classifier type can be specified to be used on the X2 feature set
		 if desired.

	k - (Optional) The number of iterations
		The default is 30 (from paper)

	u - (Optional) The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 (from paper)
	"""

	def __init__(self, prob_tensor: np.ndarray, clf, clf2, p, n, k, u, num_classes):
		self.clf1_ = clf

		#we will just use a copy of clf (the same kind of classifier) if clf2 is not specified
		if clf2 == None:
			self.clf2_ = copy.copy(clf)
		else:
			self.clf2_ = clf2

		assert p > 0 and n > 0
		self.p_ = p
		self.n_ = n
		self.k_ = k
		self.u_ = u
		self.num_classes_ = num_classes
		assert prob_tensor.shape == (num_classes, num_classes, num_classes)
		self.prob_tensor = prob_tensor
		self.cond_tensor = compute_conditionals(prob_tensor)

		random.seed()


	def fit(self, X1, X2, y):
		"""
		Description:
		fits the classifiers on the partially labeled data, y.

		Parameters:
		X1 - array-like (n_samples, n_features_1): first set of features for samples
		X2 - array-like (n_samples, n_features_2): second set of features for samples
		y - array-like (n_samples): labels for samples, -1 indicates unlabeled

		"""

		#we need y to be a numpy array so we can do more complex slicing
		y = np.asarray(y)

		#set the n and p parameters if we need to

		#the set of unlabeled samples
		U = [i for i, y_i in enumerate(y) if y_i == -1]

		#we randomize here, and then just take from the back so we don't have to sample every time
		random.shuffle(U)

		#this is U' in paper
		U_ = U[-min(len(U), self.u_):]

		#the samples that are initially labeled
		L = [i for i, y_i in enumerate(y) if y_i != -1]

		#remove the samples in U_ from U
		U = U[:-len(U_)]


		it = 0 #number of cotraining iterations we've done so far

		#loop until we have assigned labels to everything in U or we hit our iteration break condition
		while it != self.k_ and U:
			it += 1

			self.clf1_.fit(X1[L], y[L])
			self.clf2_.fit(X2[L], y[L])

			y1_prob = self.clf1_.predict_proba(X1[U_])
			y2_prob = self.clf2_.predict_proba(X2[U_])

			new_labels = {}
			for i in range(self.num_classes_):
				new_labels[i] = set()
				for j in (y1_prob[:,i].argsort())[-self.n_:]:
					if y1_prob[j,i] > 0.5:
						new_labels[i].add(j)
				for j in (y2_prob[:,i].argsort())[-self.p_:]:
					if y2_prob[j,i] > 0.5:
						new_labels[i].add(j)
			
			for i in range(self.num_classes_):
				y[[U_[x] for x in new_labels[i]]] = i
				L.extend([U_[x] for x in new_labels[i]])
			new_labels_all = set()
			for i in range(self.num_classes_):
				new_labels_all.update(new_labels[i])
			U_ = [elem for j, elem in enumerate(U_) if not (j in new_labels_all)]				

			num_to_add = sum(len(new_labels_class) for new_labels_class in new_labels.values())
			num_to_add = min(num_to_add, len(U))
			U_.extend(U[-num_to_add:])
			U = U[:-num_to_add]


		#let's fit our final model
		self.clf1_.fit(X1[L], y[L])
		self.clf2_.fit(X2[L], y[L])


	def supports_proba(self, clf):
		"""Checks if a given classifier supports the 'predict_proba' method, given a single vector x"""
		return hasattr(clf, 'predict_proba')

	def predict(self, X1, X2):
		"""
		Predict the classes of the samples represented by the features in X1 and X2.

		Parameters:
		X1 - array-like (n_samples, n_features1)
		X2 - array-like (n_samples, n_features2)


		Output:
		y - array-like (n_samples)
			These are the predicted classes of each of the samples.  If the two classifiers, don't agree, we try
			to use predict_proba and take the classifier with the highest confidence and if predict_proba is not implemented, then we randomly
			assign either 0 or 1.  We hope to improve this in future releases.

		"""

		proba_supported = supports_proba(self.clf1_) and supports_proba(self.clf2_)
		# doesn't work without predict_proba
		assert proba_supported

		y1 = self.clf1_.predict_proba(X1)
		y2 = self.clf2_.predict_proba(X2)

		no_post = y1 + y2
		no_post_pred = np.argmax(no_post, axis=1)

		#fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree
		y_pred = np.asarray([-1] * X1.shape[0])
		num_flipped = 0

		for i in range(X1.shape[0]):
			post = compute_posteriors(self.cond_tensor, y1[i], y2[i])
			y_pred[i] = np.argmax(post)
			num_flipped += (no_post_pred[i] != y_pred[i])
			


		#check that we did everything right
		assert not (-1 in y_pred)

		print('Fraction flipped: ', num_flipped / X1.shape[0])
		return y_pred

class DistributionAwareTrain(DistributionAwarePred):
	"""
	Parameters:
	clf - The classifier that will be used in the cotraining algorithm on the X1 feature set
		(Note a copy of clf will be used on the X2 feature set if clf2 is not specified).

	prob_tensor: 3D tensor of shape (n_classes, n_classes, n_classes), which gives the
	distribution of sample classes

	clf2 - (Optional) A different classifier type can be specified to be used on the X2 feature set
		 if desired.

	k - (Optional) The number of iterations
		The default is 30 (from paper)

	u - (Optional) The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 (from paper)
	"""

	def __init__(self, prob_tensor: np.ndarray, clf, clf2, p, n, k, u, num_classes):
		super().__init__(prob_tensor, clf, clf2, p, n, k, u, num_classes)
		self.one_view_conds = [compute_conditionals_view_to_view(prob_tensor, view) for view in [0, 1]]
		self.prob_tensor = prob_tensor
		self.one_view_probs = [compute_prob_one_view(prob_tensor, view) for view in [0, 1]]
		print('one view probs: ', self.one_view_probs)


 
	def fit(self, X1, X2, y1, y2, y1_true: np.ndarray, y2_true: np.ndarray, y_true: np.ndarray):
		"""
		Description:
		fits the classifiers on the partially labeled data, y.

		Parameters:
		X1 - array-like (n_samples, n_features_1): first set of features for samples
		X2 - array-like (n_samples, n_features_2): second set of features for samples
		y - array-like (n_samples): labels for samples, -1 indicates unlabeled

		"""

		#we need y to be a numpy array so we can do more complex slicing
		y1 = np.array(y1)
		y2 = np.array(y2)

		#set the n and p parameters if we need to

		#the set of unlabeled samples
		U1 = [i for i, y_i in enumerate(y1) if y_i == -1]
		U2 = [i for i, y_i in enumerate(y2) if y_i == -1]

		#we randomize here, and then just take from the back so we don't have to sample every time
		random.Random(1).shuffle(U1)
		random.Random(1).shuffle(U2)

		#this is U' in paper
		U1_ = U1[-min(len(U1), self.u_):]
		U2_ = U2[-min(len(U2), self.u_):]

		#the samples that are initially labeled
		L1 = [i for i, y_i in enumerate(y1) if y_i != -1]
		L2 = [i for i, y_i in enumerate(y2) if y_i != -1]

		#remove the samples in U_ from U
		U1 = U1[:-len(U1_)]
		U2 = U2[:-len(U2_)]


		it = 0 #number of cotraining iterations we've done so far

		#loop until we have assigned labels to everything in U or we hit our iteration break condition
		while it != self.k_ and U1 and U2:
			it += 1

			self.clf1_.fit(X1[L1], y1[L1])
			self.clf2_.fit(X2[L2], y2[L2])

			y1_prob = self.clf1_.predict_proba(X1[U1_])
			y2_prob = self.clf2_.predict_proba(X2[U2_])

			y_pred_from_y2 = np.zeros(y1_prob.shape)
			y_pred_from_y1 = np.zeros(y2_prob.shape)
			for i in range(len(y1_prob)):
				y_pred_from_y1[i] = compute_posteriors_one_view(self.one_view_conds[1], y1_prob[i])
				y_pred_from_y2[i] = compute_posteriors_one_view(self.one_view_conds[0], y2_prob[i])
			logits_avg_1 = y_pred_from_y1.mean(axis=0)
			logits_avg_2 = y_pred_from_y2.mean(axis=0)
			'''
			for i in range(len(y1_prob)):
				y_pred_from_y1[i] *= (self.one_view_probs[0] / logits_avg_1)
				y_pred_from_y2[i] *= (self.one_view_probs[1] / logits_avg_2)
			'''
			print('mean logits', y_pred_from_y1.mean(axis=0), y_pred_from_y2.mean(axis=0))

			num_flipped = 0
			new_labels_1 = {i: set() for i in range(self.num_classes_)}
			new_labels_2 = {i: set() for i in range(self.num_classes_)}
			n_select = (self.n_ + self.p_)
			for cl in range(self.num_classes_):
				for j in (y_pred_from_y1[:,cl].argsort()[-int(self.one_view_probs[0][cl] * n_select):]):
					logits = y_pred_from_y1[j]
					#pred_class = np.random.choice(self.num_classes_, p=logits)
					pred_class = logits.argmax()
					if pred_class != y1_prob[j].argmax():
						num_flipped += 1
					new_labels_2[pred_class].add(j)
				for j in (y_pred_from_y2[:,cl].argsort()[-int(self.one_view_probs[1][cl] * n_select):]):
					logits = y_pred_from_y2[j]
					#pred_class = np.random.choice(self.num_classes_, p=logits)
					pred_class = logits.argmax()
					if pred_class != y2_prob[j].argmax():
						num_flipped += 1
					new_labels_1[pred_class].add(j)
			'''
			print('view 1')
			for k, v in new_labels_1.items():
				print('class', k, 'count', len(v))
			print('view 2')
			for k, v in new_labels_2.items():
				print('class', k, 'count', len(v))
			'''
			
			changed_indices_1 = []
			changed_indices_2 = []
			for i in range(self.num_classes_):
				y1[[U1_[x] for x in new_labels_1[i]]] = i
				changed_indices_1.extend(new_labels_1[i])
				y2[[U2_[x] for x in new_labels_2[i]]] = i
				changed_indices_2.extend(new_labels_2[i])
				L1.extend([U1_[x] for x in new_labels_1[i]])
				L2.extend([U2_[x] for x in new_labels_2[i]])
			changed_indices_expanded_1 = [U1_[x] for x in changed_indices_1]
			changed_indices_expanded_2 = [U2_[x] for x in changed_indices_2]
			print('dist aware iter', it, 'accuracy y2->y1', (y1[changed_indices_expanded_1] == y1_true[changed_indices_expanded_1]).mean(), 'accuracy y1->y2', (y2[changed_indices_expanded_2] == y2_true[changed_indices_expanded_2]).mean())
			print('view 1 accuracy y1+y2->y_true', (((y_pred_from_y1[changed_indices_2] + y1_prob[changed_indices_2]) / 2).argmax(axis=1) == y_true[changed_indices_expanded_2]).mean())
			print('view 2 accuracy y1+y2->y_true', (((y_pred_from_y2[changed_indices_1] + y2_prob[changed_indices_1]) / 2).argmax(axis=1) == y_true[changed_indices_expanded_1]).mean())
			new_labels_all = set()
			for i in range(self.num_classes_):
				new_labels_all.update(new_labels_1[i])
				new_labels_all.update(new_labels_2[i])
			U1_ = [elem for j, elem in enumerate(U1_) if not (j in new_labels_all)]				
			U2_ = [elem for j, elem in enumerate(U2_) if not (j in new_labels_all)]

			num_to_add = len(new_labels_all)
			num_to_add = min(num_to_add, len(U1), len(U2))
			U1_.extend(U1[-num_to_add:])
			U2_.extend(U2[-num_to_add:])
			U1 = U1[:-num_to_add]
			U2 = U2[:-num_to_add]
			print('Training fraction flipped: ', num_flipped / (self.n_ + self.p_))


		#let's fit our final model
		self.clf1_.fit(X1[L1], y1[L1])
		self.clf2_.fit(X2[L2], y2[L2])

	def predict(self, X1, X2):
		"""
		Predict the classes of the samples represented by the features in X1 and X2.

		Parameters:
		X1 - array-like (n_samples, n_features1)
		X2 - array-like (n_samples, n_features2)


		Output:
		y - array-like (n_samples)
			These are the predicted classes of each of the samples.  If the two classifiers, don't agree, we try
			to use predict_proba and take the classifier with the highest confidence and if predict_proba is not implemented, then we randomly
			assign either 0 or 1.  We hope to improve this in future releases.

		"""

		y1 = self.clf1_.predict(X1)
		y2 = self.clf2_.predict(X2)

		proba_supported = supports_proba(self.clf1_) and supports_proba(self.clf2_)

		#fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree
		y_pred = np.asarray([-1] * X1.shape[0])

		for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
			if y1_i == y2_i:
				y_pred[i] = y1_i
			elif proba_supported:
				y1_probs = np.array(self.clf1_.predict_proba([X1[i]])[0])
				y2_probs = np.array(self.clf2_.predict_proba([X2[i]])[0])
				sum_probs = y1_probs + y2_probs
				pred = sum_probs.argmax()
				y_pred[i] = pred

			else:
				#the classifiers disagree and don't support probability, so we guess
				y_pred[i] = random.randint(0, 1)


		#check that we did everything right
		assert not (-1 in y_pred)

		return y_pred