from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
import pyspark.sql.types as types
import datetime
from pyspark.sql import Row
from pyspark.sql import DataFrameWriter
import random
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator,RegressionEvaluator
import ordereddict
import sys

spark = SparkSession.builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.local.ip", "10.2.0.15") \
    .getOrCreate()    

n_features = 4
n_trees = 10    
    
print ""

def build_model(algo, df, n_features):

    used_features = []    
    for i in range(n_features):
        used_features.append('col_'+str(i))

    vectorAssembler = VectorAssembler(
        inputCols=[x for x in df.columns if x in used_features],
        outputCol='features')
        
    df2 = vectorAssembler.transform(df)
    t1 = datetime.datetime.now()
    model = algo.fit(df2)
    td = model.transform(df2)
    t2 = datetime.datetime.now()
    train_time = t2-t1
    print('fit + transform time    \t %s' % ((train_time).__str__()))
    return (model,td)
    
def calc_conf_matrix(td, classes, printout=True):
    nrows = td.count()
    confMatrix=td.groupBy("label", "prediction").count().collect()
    confMatrix=[(elem[0],elem[1],elem[2]) for elem in confMatrix]
    if printout:
        print(confMatrix)
    if len(classes)==2:
        c = classes[1]
        tp = float([t[2] for t in confMatrix if t[0] == c and t[1] == c][0])
        tn = float([t[2] for t in confMatrix if t[0] != c and t[1] != c][0])

        fp = float([t[2] for t in confMatrix if t[0] != c and t[1] == c][0])
        fn = float([t[2] for t in confMatrix if t[0] == c and t[1] != c][0])
        cm = [[tp,fp],[fn,tn]]
        spc = tn / (fp + tn)    # specificity
        ppv = tp / (tp + fp)    # precision
        npv = tn / (tn + fn)    # negative predictive value
        fpr = fp / (fp + tn)    # false positive rate 
        fdr = fp / (fp + tp)    # false discovery rate 
        fnr = fn / (fn + tp)    # false negative rate 
        acc = (tp + tn) / (nrows)   #accuracy 
        tpr = tp / (tp + fn)      # recall 
        f1 = 2*tp / (2*tp + fp + fn)  # f1 score 
        if printout:
            print("  specificity=              %f\n"
                  "  precision=                %f\n"
                  "  negative predictive value=%f\n"
                  "  false positive rate=      %f\n"
                  "  false discovery rate =    %f\n"
                  "  false negative rate=      %f\n"
                  "  accuracy=                 %f\n"
                  "  recall=                   %f\n"
                  "  f1 score=                 %f" % (spc, ppv, npv, fpr, fdr, fnr, acc, tpr, f1))
        return cm,ppv,tpr,acc
    else:
        tp = ordereddict.OrderedDict()
        tn = ordereddict.OrderedDict()
        fp = ordereddict.OrderedDict()
        fn = ordereddict.OrderedDict()
        for c in classes:
            tp[c] = float([t[2] for t in confMatrix if t[0] == c and t[1] == c][0])
            tn[c] = float([t[2] for t in confMatrix if t[0] != c and t[1] != c][0])

            fp[c] = float([t[2] for t in confMatrix if t[0] != c and t[1] == c][0])
            fn[c] = float([t[2] for t in confMatrix if t[0] == c and t[1] != c][0])
            spc = tn[c] / (fp[c] + tn[c])    # specificity
            ppv = tp[c] / (tp[c] + fp[c])    # precision
            npv = tn[c] / (tn[c] + fn[c])    # negative predictive value
            fpr = fp[c] / (fp[c] + tn[c])    # false positive rate 
            fdr = fp[c] / (fp[c] + tp[c])    # false discovery rate 
            fnr = fn[c] / (fn[c] + tp[c])    # false negative rate 
            acc = (tp[c] + tn[c]) / (nrows)   #accuracy 
            acc = (tp[c] + tn[c]) / (nrows)   #accuracy 
            f1 = 2*tp[c] / (2*tp[c] + fp[c] + fn[c])  # f1 score 
            if printout:
                print("  class              %f\n"
                      "  specificity=              %f\n"
                      "  precision=                %f\n"
                      "  negative predictive value=%f\n"
                      "  false positive rate=      %f\n"
                      "  false discovery rate =    %f\n"
                      "  false negative rate=      %f\n"
                      "  accuracy=                 %f\n"
                      "  f1 score=                 %f" % (c, spc, ppv, npv, fpr, fdr, fnr, acc, f1))
        
def calc_binary_metrics(model, algo, td, classes, printout=False):
    # Coefficients 
    if type(model).__name__ == 'NaiveBayesModel' or type(model).__name__ == 'RandomForestClassificationModel' or type(model).__name__ == 'DecisionTreeClassificationModel' or type(model).__name__ == 'GBTClassificationModel':
        coefficients = 'coefficients are not available for RandomForestClassification, DecisionTreeClassification, GBTClassification and NaiveBayesModel'
    else:
        coefficients = model.coefficients
    # Confusion matrix 
        # Precision 
        # Recall 
        # Accuracy 
    confMatrix,ppv,tpr,acc = calc_conf_matrix(td, classes, printout=printout)

    # AUC 
    eval = BinaryClassificationEvaluator()
    areaUnderROC = eval.evaluate(td,{'metricName':'areaUnderROC'})        
    areaUnderPR = eval.evaluate(td,{'metricName':'areaUnderPR'})        

    # ROC 
    if type(model).__name__ == 'LogisticRegressionModel' or type(model).__name__ == 'GeneralizedLinearRegressionModel':
        roc = model.summary.roc.collect()
    else:
        roc = 'ROC is available only for LogisticRegression and GeneralizedLinearRegression'
    if not printout:
        print("  coefficients =            %s\n"
              "  Confusion matrix=         %s\n"
              "  precision=                %f\n"
              "  recall=                   %f\n"
              "  accuracy=                 %f\n"
              % (coefficients, confMatrix, ppv, tpr, acc))    
    else:
        print("  areaUnderROC=             %f\n"
              "  areaUnderPR=              %f\n"
              "  ROC=                      %s"
              % (areaUnderROC, areaUnderPR, roc))    
    #-----------------------------------------------
    # Explained sum of squares
    # P-values -------------?????????

def calc_multiclass_metrics(model, algo, td, classes, printout=False):
    calc_conf_matrix(td, classes, printout)
    
    eval = MulticlassClassificationEvaluator()

    accuracy = eval.evaluate(td,{'metricName':'accuracy'})
    f1 = eval.evaluate(td,{'metricName':'f1'})
    weightedPrecision = eval.evaluate(td,{'metricName':'weightedPrecision'})
    weightedRecall = eval.evaluate(td,{'metricName':'weightedRecall'})

    print("  accuracy=          %f\n"
          "  f1=                %f\n"
          "  weightedPrecision= %f\n"
          "  weightedRecall=    %f" %(accuracy, f1, weightedPrecision, weightedRecall))
    
def calc_regression_metrics(model, algo, td, classes, printout=False):
    eval = RegressionEvaluator()
    rmse = eval.evaluate(td,{'metricName':'rmse'})
    mse = eval.evaluate(td,{'metricName':'mse'})
    r2 = eval.evaluate(td,{'metricName':'r2'})
    mae = eval.evaluate(td,{'metricName':'mae'})

    print(' rmse=%f\n mse=%f\n r2=%f\n mae=%f' %(rmse, mse, r2, mae))
    if type(model).__name__ == 'GeneralizedLinearRegressionModel' or type(model).__name__ == 'LinearRegressionModel':
        if type(model).__name__ == 'GeneralizedLinearRegressionModel':
            print("  aic=                       %f\n"
                  "  deviance=                  %f\n"
                  "  nullDeviance=              %f\n"
                  "  dispersion=                %f\n"
                  "  rank=                      %d\n"
                  "  residualDegreeOfFreedom=   %d\n"
                  "  residualDegreeOfFreedomNull= %d\n"
                  %( model.summary.aic, model.summary.deviance, model.summary.nullDeviance, model.summary.dispersion,
                model.summary.rank, model.summary.residualDegreeOfFreedom,
                model.summary.residualDegreeOfFreedomNull))
        elif type(model).__name__ == 'LinearRegressionModel':
            print("  rootMeanSquaredError=      %f\n"
                  "  meanAbsoluteError=         %f\n"
                  "  devianceResiduals=         %s\n"
                  "  meanSquaredError=          %f\n"
                  "  explainedVariance=         %f"
                  %(model.summary.rootMeanSquaredError, model.summary.meanAbsoluteError, model.summary.devianceResiduals,
                  model.summary.meanSquaredError, model.summary.explainedVariance))
        print("  rootMeanSquaredError=      %s\n"
              "  degreesOfFreedom=          %d\n"
              "  pValues=                   %s\n"
              "  residuals=                 [...]\n"
              "  tValues=                   %s"
              %( model.summary.coefficientStandardErrors, model.summary.degreesOfFreedom,
              model.summary.pValues, model.summary.tValues))
                
def calc_clustering_metrics(model, algo, td, classes):
    if type(model).__name__ == 'BisectingKMeansModel' or type(model).__name__ == 'KMeansModel':
        print("  n_clusters =              %d\n"
              "  cluster centers=          %s\n"
              "  cluster sizes=            %s"
              % (model.summary.k, model.clusterCenters(), model.summary.clusterSizes))    
    elif type(model).__name__ == 'GaussianMixtureModel':
        print("  n_clusters =              %d\n"
              "  cluster sizes=            %s\n"
              "  logLikelihood=            %f\n"
              "  weights=                  %s\n"
              "  mean=                     %s\n"
              "  cov=                      %s"
              % (model.summary.k, model.summary.clusterSizes, model.summary.logLikelihood, model.weights,
              model.gaussiansDF.select("mean").collect(),model.gaussiansDF.select("cov").collect()))    
    
    
def calc_metrics(model, algo, td, family, classes, print_conf_matrix=False):
    t1 = datetime.datetime.now()
    if family == 'classification':
        if len(classes) == 2:
            calc_binary_metrics(model, algo, td, classes, printout=print_conf_matrix)
        else:
            calc_multiclass_metrics(model, algo, td, classes, printout=print_conf_matrix)
    elif family == 'regression':
        calc_regression_metrics(model, algo, td, classes, printout=print_conf_matrix)
    elif family == 'clustering':
        calc_clustering_metrics(model, algo, td, classes)

    t2 = datetime.datetime.now()
    metrics_time = t2-t1
    print('metrics calculation time\t %s' % ((metrics_time).__str__()))
 
def test_model(algo, df, n_features, print_conf_matrix):
    family = str(type(algo)).split('.')[2]
    classes = ordereddict.OrderedDict()
    if family == 'classification':
        classes = [l.label for l in df.select('label').distinct().collect()]
        classes.sort()

        print('\n%s - %s, %d classes, %d features' % ((str(algo).split('_')[0]),family, len(classes), n_features))
    elif family == 'regression':
        print('\n%s - %s' % ((str(algo).split('_')[0]),family))
    elif family == 'clustering':
        print('\n%s - %s' % ((str(algo).split('_')[0]),family))
    else:
        raise Exception('Only regression or classification are supported, got "%s"' % family)

    (model,td) = build_model(algo, df, n_features)
    calc_metrics(model, algo, td, family, classes, print_conf_matrix)
    return (model,td)
###########################################################
# Binary classifications
df_binary = spark.read.parquet('../Dataframes/small_df_1000_10')
df_multi = spark.read.parquet('../Dataframes/df_1000x10_4classes')
df_clusters = spark.read.parquet('../Dataframes/clusters_1000x4')

n_features = 4


print("\n Binary classifiers df: %dx%d" % (len(df_binary.columns), df_binary.count()))
print("\n n_features: %d" % n_features)

# 1. logistic reg 
from pyspark.ml.classification import LogisticRegression
algo = LogisticRegression()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=False)

# 2. svm 
from pyspark.ml.classification import LinearSVC
algo = LinearSVC()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=False)

# 3. random forest 
from pyspark.ml.classification import RandomForestClassifier
algo = RandomForestClassifier()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=False)

# 4. decision tree 
from pyspark.ml.classification import DecisionTreeClassifier
algo = DecisionTreeClassifier()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=False)

# 5. decision tree 
from pyspark.ml.classification import GBTClassifier
algo = GBTClassifier()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=False)

# 6. decision tree 
from pyspark.ml.clustering import GaussianMixture
algo = GaussianMixture()
(model,td) = test_model(algo, df_clusters, n_features, print_conf_matrix=False)


print('===========================================================================')
print('==   classifiers binary  ==================================================')
print('===========================================================================')
print("\n n_features: %d" % n_features)

from pyspark.ml.regression import GeneralizedLinearRegression
algo = GeneralizedLinearRegression(family='binomial')
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=True)

from pyspark.ml.classification import NaiveBayes
algo = NaiveBayes()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=True)


print('===========================================================================')
print('==   classifiers multiclass  ==============================================')
print('===========================================================================')
print("\n n_features: %d" % n_features)

from pyspark.ml.classification import LogisticRegression
algo = LogisticRegression()
(model,td) = test_model(algo, df_multi, n_features, print_conf_matrix=True)

from pyspark.ml.classification import DecisionTreeClassifier
algo = DecisionTreeClassifier()
(model,td) = test_model(algo, df_multi, n_features, print_conf_matrix=True)
# rawPrediction=DenseVector([72.0, 93.0]), probability=DenseVector([0.4364, 0.5636]), prediction=1.0

from pyspark.ml.classification import RandomForestClassifier
algo = RandomForestClassifier()
(model,td) = test_model(algo, df_multi, n_features, print_conf_matrix=True)

from pyspark.ml.classification import NaiveBayes
algo = NaiveBayes()
(model,td) = test_model(algo, df_multi, n_features, print_conf_matrix=True)


print('===========================================================================')
print('==   regressors  ==========================================================')
print('===========================================================================')
print("\n n_features: %d" % n_features)

from pyspark.ml.regression import DecisionTreeRegressor
algo = DecisionTreeRegressor()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=True)

from pyspark.ml.regression import GBTRegressor
algo = GBTRegressor()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=True)

from pyspark.ml.regression import GeneralizedLinearRegression
algo = GeneralizedLinearRegression()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=True)

from pyspark.ml.regression import IsotonicRegression
algo = IsotonicRegression()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=True)

from pyspark.ml.regression import LinearRegression
algo = LinearRegression()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=True)

from pyspark.ml.regression import RandomForestRegressor
algo = RandomForestRegressor()
(model,td) = test_model(algo, df_binary, n_features, print_conf_matrix=True)

print('===========================================================================')
print('==   Clustering  ==========================================================')
print('===========================================================================')
print("\n n_features: %d" % n_features)

from pyspark.ml.clustering import KMeans
algo = KMeans()
(model,td) = test_model(algo, df_clusters, n_features, print_conf_matrix=False)

from pyspark.ml.clustering import BisectingKMeans
algo = BisectingKMeans()
(model,td) = test_model(algo, df_clusters, n_features, print_conf_matrix=False)

