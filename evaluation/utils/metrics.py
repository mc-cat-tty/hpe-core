
import numpy as np


class _SamplesAccumulator:
    """
    Base class for metrics accumulating samples over multiple calls of update_samples(); samples are stored as a list of
    chunks and concatenated only when accessed, since concatenating at every update (np.append) would copy all the data
    accumulated so far, with quadratic cost in the number of updates
    """

    _fields = ('predicted_joints', 'gt_joints')

    def __init__(self):
        self._chunks = {field: [] for field in self._fields}

    def _accumulate(self, **arrays) -> None:
        for field, array in arrays.items():
            self._chunks[field].append(array)

    def _get_samples(self, field: str):
        chunks = self._chunks[field]
        if not chunks:
            return None
        if len(chunks) > 1:
            chunks[:] = [np.concatenate(chunks, axis=0)]
        return chunks[0]

    @property
    def predicted_joints(self):
        return self._get_samples('predicted_joints')

    @property
    def gt_joints(self):
        return self._get_samples('gt_joints')


class PCK(_SamplesAccumulator):
    """
    Class for computing HPE metric PCK (Percentage of Correct Keypoints)
    """

    _fields = ('predicted_joints', 'gt_joints', 'reference_sizes')

    def __init__(self, threshold: float):
        """
        Class initializer.
        Parameters:
            threshold (float): threshold value used for classifying joints as correct or not correct; must be > 0.0 and <= 1.0
        """

        assert .0 < threshold <= 1.0, 'PCK threshold value must be > 0.0 and <= 1.0'

        super().__init__()
        self.threshold = threshold

    @property
    def reference_sizes(self):
        return self._get_samples('reference_sizes')

    def update_samples(self, predicted_joints: np.array, gt_joints: np.array, reference_sizes: np.array) -> None:
        """
        Accumulates data that will be used for computing PCK (with function get_value()).
        It expects predicted_joints and gt_joints with shape [batch_size, joints_num, 2 or 3] and
        not annotated joints as [-1, -1] or [-1, -1, -1]
        Parameters:
            predicted_joints (numpy.array): predicted joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
            gt_joints (numpy.array): ground truth joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
            reference_sizes (numpy.array): sizes of a body part (either head or torso, used for classifying joints as correct or not correct) as a numpy array with shape [batch_size, joints_num]
        """
        self._accumulate(predicted_joints=predicted_joints, gt_joints=gt_joints, reference_sizes=reference_sizes)

    def get_value(self):
        """
        Computes metric using the accumulated data.
        Return:
            numpy.array: PCK values for each single joint
            float: average of PCK values computed over all joints
        """

        if self.reference_sizes is None:
            return np.zeros(1, dtype=float), .0

        # compute PCK's threshold as percentage of head size in pixels for each pose
        thresholds_head = self.reference_sizes * self.threshold
        # thresholds_head = thresholds_head.reshape([-1, 1]).tile((1, self.gt_joints.shape[1]))
        thresholds_head = np.tile(thresholds_head.reshape([-1, 1]), (1, self.gt_joints.shape[1]))

        # compute euclidean distances between joints
        distances = np.linalg.norm(self.predicted_joints - self.gt_joints, axis=2)

        # compute correct keypoints; non detected joints (both coordinates equal to 0) are always incorrect, even if
        # the GT joint lies within the threshold from the image origin
        correct_keypoints = np.logical_and(distances <= np.array(thresholds_head),
                                           pred_detected(self.predicted_joints)).astype(int)

        # remove not annotated keypoints from pck computation
        annotated = gt_annotated(self.gt_joints)
        correct_keypoints = correct_keypoints * annotated.astype(int)
        annotated_keypoints_num = np.sum(annotated.astype(int), axis=0)

        # compute pck
        pck_joints = np.sum(correct_keypoints, axis=0) / annotated_keypoints_num
        pck_avg = np.mean(pck_joints)

        return pck_joints, pck_avg

    def get_weights(self):
        """
        Returns the number of samples the PCK of each joint is computed on, i.e. the number of annotated GT joints
        """
        return np.sum(gt_annotated(self.gt_joints), axis=0)


class RMSE(_SamplesAccumulator):

    def __init__(self):
        super().__init__()

    def update_samples(self, predicted_joints: np.array, gt_joints: np.array) -> None:
        """
        Accumulates data that will be used for computing RSME (with function get_value()).
        It expects predicted_joints and gt_joints with shape [batch_size, joints_num, 2 or 3] and
        not annotated joints as [-1, -1] or [-1, -1, -1]
        Parameters:
            predicted_joints (numpy.array): predicted joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
            gt_joints (numpy.array): ground truth joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
        """
        self._accumulate(predicted_joints=predicted_joints, gt_joints=gt_joints)

    def get_value(self):
        """
        Computes metric using the accumulated data.
        Return:
            numpy.array: normalized RMSE values computed on x and y coordinates of each single joint
            numpy.array: average of normalized RMSE values computed on x and y coordinates of all joints
            numpy.array: maximum RMSE values between the x and y coordinates of all joints
        """

        rmse = np.sqrt(np.sum(np.square(self.predicted_joints - self.gt_joints), axis=0) / len(self.gt_joints))
        normalized_rmse = (rmse / abs(np.mean(self.predicted_joints, axis=0))) * 100
        normalized_rmse_avg = np.mean(normalized_rmse, axis=0)
        max_normalized_rmse = np.max(normalized_rmse, axis=1)

        return normalized_rmse.flatten(), normalized_rmse_avg.flatten(), max_normalized_rmse

class MPJPE(_SamplesAccumulator):
    def __init__(self):
        super().__init__()
        
    def update_samples(self, predicted_joints: np.array, gt_joints: np.array) -> None:
        """
        Accumulates data that will be used for computing MPJPE (with function get_value()).
        It expects predicted_joints and gt_joints with shape [batch_size, joints_num, 2 or 3] and
        not annotated joints as [-1, -1] or [-1, -1, -1]
        Parameters:
            predicted_joints (numpy.array): predicted joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
            gt_joints (numpy.array): ground truth joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
        """
        self._accumulate(predicted_joints=predicted_joints, gt_joints=gt_joints)

        
    def get_value(self):
        """
        Computes MPJPE (Mean Per Joint Position Error)
        """
    
        # compute euclidean distances between joints
        distances = np.linalg.norm(self.predicted_joints - self.gt_joints, axis=2)

        # compute errors without considering non annotated GT joints and non detected joints (both coordinates
        # equal to 0); non detected joints are tracked by metric MissedDetections
        valid = np.logical_and(gt_annotated(self.gt_joints), pred_detected(self.predicted_joints))
        res = np.sum(distances * valid, axis=0) / np.sum(valid, axis=0)

        mpje_avg = np.mean(res)

        return res.flatten(), mpje_avg

    def get_weights(self):
        """
        Returns the number of samples the MPJPE of each joint is computed on, i.e. the number of annotated GT joints
        whose prediction has been detected
        """
        return np.sum(np.logical_and(gt_annotated(self.gt_joints), pred_detected(self.predicted_joints)), axis=0)


class MissedDetections(_SamplesAccumulator):
    """
    Class for computing the percentage of missed detections, i.e. annotated GT joints whose prediction has both
    coordinates equal to 0
    """

    def __init__(self):
        super().__init__()

    def update_samples(self, predicted_joints: np.array, gt_joints: np.array) -> None:
        """
        Accumulates data that will be used for computing the missed detections (with function get_value()).
        It expects predicted_joints and gt_joints with shape [batch_size, joints_num, 2 or 3] and
        not annotated joints as [-1, -1] or [-1, -1, -1]
        Parameters:
            predicted_joints (numpy.array): predicted joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
            gt_joints (numpy.array): ground truth joints as a numpy array with shape [batch_size, joints_num, 2 or 3]
        """
        self._accumulate(predicted_joints=predicted_joints, gt_joints=gt_joints)

    def get_value(self):
        """
        Computes metric using the accumulated data.
        Return:
            numpy.array: percentage of missed detections for each single joint
            float: average of the percentages computed over all joints
        """

        annotated = gt_annotated(self.gt_joints)
        missed = np.logical_and(annotated, np.logical_not(pred_detected(self.predicted_joints)))
        missed_joints = np.sum(missed, axis=0) / np.sum(annotated, axis=0) * 100
        missed_avg = np.mean(missed_joints)

        return missed_joints, missed_avg

    def get_weights(self):
        """
        Returns the number of samples the missed detections of each joint are computed on, i.e. the number of
        annotated GT joints
        """
        return np.sum(gt_annotated(self.gt_joints), axis=0)


class SequencesWeightedStd:
    """
    Class for computing the standard deviation of a metric (PCK, MPJPE or MissedDetections) over sequences; each
    sequence is weighted by the number of samples its metric is computed on, so that the weighted mean of the
    per-sequence values is the metric computed on the samples pooled over all sequences
    """

    def __init__(self):
        self._values = []
        self._weights = []

    def update_samples(self, joints_values: np.array, joints_weights: np.array) -> None:
        """
        Accumulates the metric computed on a single sequence.
        Parameters:
            joints_values (numpy.array): metric values for each single joint, with shape [joints_num]
            joints_weights (numpy.array): number of samples the metric of each joint is computed on (metric's
                                          get_weights()), with shape [joints_num]
        """
        self._values.append(np.asarray(joints_values, dtype=float))
        self._weights.append(np.asarray(joints_weights, dtype=float))

    def get_value(self):
        """
        Computes the weighted standard deviations over the accumulated sequences.
        Return:
            numpy.array: standard deviation of the metric of each single joint
            float: standard deviation of the metric averaged over all joints
        """

        weights = np.stack(self._weights)
        # joints without samples in a sequence have an undefined (nan) value, and no weight
        values = np.where(weights > 0, np.stack(self._values), 0.)
        std_joints = _weighted_std(values, weights)

        # average over the joints with samples of each sequence, weighted by the total samples of the sequence
        valid_joints_num = np.sum(weights > 0, axis=1)
        avg_values = np.divide(np.sum(values, axis=1), valid_joints_num, out=np.zeros(len(values)),
                               where=valid_joints_num > 0)
        std_avg = _weighted_std(avg_values, np.sum(weights, axis=1))

        return std_joints, std_avg


def _weighted_std(values: np.array, weights: np.array) -> np.array:
    """
    Computes the weighted (population) standard deviation of values along the first axis
    """
    total_weights = np.sum(weights, axis=0)
    mean = np.sum(values * weights, axis=0) / total_weights
    return np.sqrt(np.sum(weights * np.square(values - mean), axis=0) / total_weights)


def gt_annotated(gt_joints: np.array) -> np.array:
    """
    Returns a boolean mask with shape [batch_size, joints_num], True where the GT joint is annotated
    """
    return np.all(gt_joints > 0, axis=2)


def pred_detected(predicted_joints: np.array) -> np.array:
    """
    Returns a boolean mask with shape [batch_size, joints_num], False where the predicted joint has all coordinates
    equal to 0 (i.e. the joint has not been detected)
    """
    return np.any(predicted_joints != 0, axis=2)
    
    # def print_mpjpe(mpjpe, keypoints_str):
    #     for ei, error in enumerate(mpjpe):
    #         print(f'{keypoints_str[ei]}: {error}')
