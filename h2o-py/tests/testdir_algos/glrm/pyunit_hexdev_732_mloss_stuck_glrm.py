from __future__ import print_function
import sys

sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator

# customer dataset use 2 categorical columns of high cardinality according to michalk.
# The code run and stuck at mloss function calculation.
# It is okay for this dataset to not have an assert statement, we want to just run and make sure we do not
# get stuck too.

def glrm_allCats():
    traindata = pyunit_utils.random_dataset_enums_only(10000, 2, factorL=5, misFrac=0, randSeed=12345)

    glrm_h2o = H2OGeneralizedLowRankEstimator(k=1, loss="Quadratic", regularization_x="non_negative", 
                                              regularization_y="non_negative", max_iterations=1000)
    glrm_h2o.train(x=traindata.names, training_frame=traindata)
    glrm_h2o.show()
    

if __name__ == "__main__":
    pyunit_utils.standalone_test(glrm_allCats)
else:
    glrm_allCats()
