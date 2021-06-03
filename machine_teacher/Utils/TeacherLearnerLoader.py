"""
This modules just loads Teachers and Learners from
a string name. It's used by the Report module, to load
objects from theirs names on configurations files
"""

from .. import Learners
from .. import Teachers

_D_LEARNERS = {
	Learners.LogisticRegressionLearner.name: Learners.LogisticRegressionLearner,
	Learners.SVMLinearLearner.name: Learners.SVMLinearLearner,
	Learners.LGBMLearner.name: Learners.LGBMLearner,
	Learners.RandomForestLearner.name: Learners.RandomForestLearner,
	Learners.DecisionTreeLearner.name: Learners.DecisionTreeLearner
}

_D_TEACHERS = {
	Teachers.WTFTeacher.name: Teachers.WTFTeacher,
	Teachers.SingleBatchTeacher.name: Teachers.SingleBatchTeacher,
	Teachers.DoubleTeacher.name: Teachers.DoubleTeacher,
	Teachers.FixedPercWrongTeacher.name: Teachers.FixedPercWrongTeacher
}

def get_teacher(teacher_name, args):
	return _D_TEACHERS[teacher_name](**args)

def get_learner(learner_name, args):
	return _D_LEARNERS[learner_name](**args)