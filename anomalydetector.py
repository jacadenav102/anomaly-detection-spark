from tranformariontools import SparkTransformationTools
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import GaussianMixture
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import row_number
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA
import pandas as pd
import numpy as np
import datetime
import sys

def fn_create_checker(threshold):
    @udf(returnType=IntegerType())
    def fn_check_probabilities(listprobability):
        if all( i < threshold for i in listprobability):
            return 1   
        else:
            return 0
    return fn_check_probabilities

class AnomalyDetector:
    def __init__(self):
        self.Silhouette = None
        self.optimalK = None
        #pass


    def fn_create_metrics_table(self, date, vPandas = None ,error = False, train = True ):
        
        
        if train is True:
            vType = 'anomalias_fricciones_ent'
        
        else:
            vType = 'anomalias_fricciones_pre'
        
        if error is False:

            vPandas['id_momento'] = date
            vPandas['proyecto'] = 'FRICCIONES'
            vPandas['modelo'] = vType
            vPandas['ejecucion'] = 1
            vPandas['id_indicador'] = 2 
            vPandas['nombre_indicador'] = vPandas['anomalies'].apply( lambda x : 'anomalias_gf_' +  str(x))
            vPandas = vPandas.drop('anomalies', axis = 1)
            
        else:
            vListMonitoring = []
            vListMonitoring.append(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            vListMonitoring.append('FRICCIONES')
            vListMonitoring.append(vType)
            vListMonitoring.append(2)
            vListMonitoring.append('anomalias_gf_1')
            vListMonitoring.append(0)
            vListMonitoring.append(None)

            vPandas = pd.DataFrame()
            vPandas = vPandas.append([vListMonitoring])
            vPandas.columns = ['id_momento', 'proyecto', 'modelo', 'id_indicador',
                                'nombre_indicador', 'ejecucion', 'valor_indicador'] 
        return vPandas

                
    def fn_save_model(self, path):
        self.model.write().overwrite().save(path)

    def fn_load_models(self, queryInputable, categoricalColumns,
                       identificationColumn, NumCluster = 3,
                        scalerPath = 'AnomalyScaler', modelPath = 'AnomalyModel', PCAPath='PCAModel'):
        tranSpark = SparkTransformationTools()
        spark = tranSpark.get_spark_session('Anomaly_detector')
        vDfbegin = spark.sql(queryInputable).limit(500000)
        vDf = vDfbegin.na.drop()
        vInference = tranSpark.fn_get_inferenceData(vDf, categoricalColumns, identificationColumn)
        vInference = vInference.withColumn('Features', col('features'))
        vScaler = StandardScaler(inputCol="Features", outputCol="scaledfeatures",
                                withStd=True, withMean=True)
        vScalerModel = vScaler.fit(vInference)
        vInference = vScalerModel.transform(vInference)
        vInference = vInference.drop('Features')
        vPca = PCA(k=4, inputCol="scaledfeatures", outputCol="features")
        vPcaModel = vPca.fit(vInference)
        vInference = vPcaModel.transform(vInference).select("features", identificationColumn)
        vFinal = self.fn_get_clusters(vInference, NumCluster)
        vModel = self.model
        vModel = vModel.load(modelPath)
        vScalerModel = vScalerModel.load(scalerPath)
        vPcaModel = vPcaModel.load(PCAPath)
        return (vModel, vScalerModel, vPcaModel)


    def fn_get_best_clusters(self, dataFrame, maximunK = 4, seed = 123):
        vListSilhouette = []
        vListKcluster = []
        evaluator = ClusteringEvaluator()
        for vK in range(3, maximunK + 1):
            try: 
                vKmeans = GaussianMixture(k=vK, seed=seed)
                model = vKmeans.fit(dataFrame)
                vPrediction = model.transform(dataFrame)
                vSilhouette = evaluator.evaluate(vPrediction)
                vListSilhouette.append(vSilhouette)
                vListKcluster.append(vK)
            except: 
                continue 

        if  not vListSilhouette:
            self.Silhouette = None
            self.optimalK = 3
            return 3
        else:
            vIndex = np.argmax(vListSilhouette)
            self.Silhouette = vListSilhouette[vIndex]
            self.optimalK = vListKcluster[vIndex]
            return vListKcluster[vIndex]

    def fn_get_clusters(self, dataFrame,
                       seed=123,
                       maximunK = 6,
                       NumCluster = 0,
                       trainedModel = None,
                       modelPath=None):
        
        if trainedModel is not None:
            self.model =  trainedModel
            vPrediction = self.model.transform(dataFrame)
            evaluator = ClusteringEvaluator()
            vDictmodel = {param[0].name: param[1] for param in trainedModel.extractParamMap().items()} 
            self.Silhouette = evaluator.evaluate(vPrediction)
            self.optimalK = vDictmodel['k']
            return vPrediction
        
        else:
        
            if NumCluster == 0:
                vNumCluster = self.fn_get_best_clusters(dataFrame=dataFrame,
                                            maximunK=maximunK,
                                            seed=seed)
            else:
                vNumCluster = NumCluster
        vGmm = GaussianMixture(k=vNumCluster, seed=seed)
        model = vGmm.fit(dataFrame)
        vPrediction = model.transform(dataFrame)
        self.model = model
        if modelPath is not None:
            model.write().overwrite().save(modelPath)
        return vPrediction
    

    def fn_get_anomalies(self ,
                         identificationColumn,
                         maximunK = 6,
                         seed = 123,
                         threshold = 0.7,
                         NumCluster = 0,
                         trainedModel = None,
                         modelPath=None,
                         Prediction=None,
                         dataFrame=None):
        
        if Prediction is None:
            vPrediction = self.fn_get_clusters(dataFrame=dataFrame,
                                            maximunK=maximunK,
                                            seed=seed,
                                            NumCluster = NumCluster,
                                            trainedModel = trainedModel,
                                            modelPath=modelPath)
        else:
            vPrediction = Prediction
        fn_check_probabilities = fn_create_checker(threshold)
        prediction = vPrediction.select('probability', fn_check_probabilities('probability').alias('anomalies')).withColumn('row_index', row_number().over(Window.orderBy(monotonically_increasing_id())))
        infereceData = vPrediction.select(identificationColumn, "prediction").withColumn('row_index', row_number().over(Window.orderBy(monotonically_increasing_id())))
        vDataresult = infereceData.join(prediction, on=["row_index"]).drop("row_index")   
        return vDataresult   

    def fn_train_model(self, queryInputable, categoricalColumns, identificationColumn, metricTable = 'proceso_enmascarado.monitoreo_frcan',
                       pipelinePath = 'anomalyPipeline' ,threshold=0.7, exitTable = 'proceso_enmascarado.frc_resultado_anomalia', seed=123):
        
        try:
            reload(sys)
            sys.setdefaultencoding('utf-8')
            vDate = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            tranSpark = SparkTransformationTools()
            spark = tranSpark.get_spark_session('anomaly_detector')
            spark.sparkContext.setLogLevel("WARN")
            vDfbegin = spark.sql(queryInputable)
            vDf = vDfbegin.na.drop()
            stages = tranSpark.fn_get_preparationStages(categoricalColumns, vDf.columns, identificationColumn)
            vScaler = StandardScaler(inputCol="Features", outputCol="scaledfeatures",
                                    withStd=True, withMean=True)
            stages.append(vScaler)
            vPca = PCA(k=6, inputCol="scaledfeatures", outputCol="features")
            stages.append(vPca)
            vPrePipeline = Pipeline(stages=stages).fit(vDf)
            vInference = vPrePipeline.transform(vDf).select(identificationColumn, "features")
            vNumCluster = self.fn_get_best_clusters(vInference, maximunK=6)
            finalStages = vPrePipeline.stages
            vGmm = GaussianMixture(k=vNumCluster , seed=seed).fit(vInference)
            finalStages.append(vGmm )
            trainedPL = PipelineModel(finalStages)
            trainedPL.write().overwrite().save(pipelinePath)
            vPrediction = vGmm.transform(vInference)
            vFinal = self.fn_get_anomalies( identificationColumn=identificationColumn, threshold=threshold, Prediction=vPrediction)
            vPercetage = vFinal.select('anomalies')
            vTotal  = vPercetage.count()
            vPercetage = vPercetage.groupBy('anomalies').count().withColumnRenamed('count', 'cnt_por_clase').withColumn('valor_indicador', (col('cnt_por_clase') / vTotal) * 100 ).sort(col('valor_indicador').desc())
            vPercetage = vPercetage.drop('cnt_por_clase')
            vPandas = vPercetage.toPandas()
            vMetricTable = self.fn_create_metrics_table(vPandas=vPandas, date=vDate, train=True)
            vSilhouetteMetrics = [self.Silhouette, vDate, 'FRICCIONES', 'anomalias_fricciones', 1, 1, 'silhouette_score']
            vClustersMetrics = [self.optimalK, vDate, 'FRICCIONES', 'anomalias_fricciones', 1, 1, 'numero_clusters']
            vLenTable = len(vMetricTable)
            vMetricTable.loc[vLenTable] = vSilhouetteMetrics
            vMetricTable.loc[vLenTable + 1 ] = vClustersMetrics
            spark.conf.set("spark.sql.execution.arrow.enabled", "true")
            schema = StructType([
            StructField('valor_indicador', FloatType(), True),
            StructField('id_momento', StringType(), False),
            StructField('proyecto', StringType(), False),
            StructField('modelo', StringType(), False),
            StructField('ejecucion', IntegerType(), False),
            StructField('id_indicador', IntegerType(), False),
            StructField('nombre_indicador', StringType(), False),])
            vMetricTable = spark.createDataFrame(vMetricTable, schema=schema)
            vMetricTable.show()
            vMetricTable.write.format("hive").option("fileFormat","parquet").saveAsTable(metricTable, mode='append')
            vFinal = vFinal.withColumn("probability",col("probability").cast(StringType())).withColumn("anomalies", col("anomalies").cast(IntegerType())).withColumn(identificationColumn,col(identificationColumn).cast(StringType()))
            vFinal.write.format("hive").option("fileFormat","parquet").saveAsTable(exitTable, mode='overwrite')
            spark.catalog.refreshTable(exitTable)

        
        except Exception as vError:
            print('Error Prediccion: {}'.format(vError))
            vMetricTable = self.fn_create_metrics_table(error=True, date=vDate, train=True)
            spark.conf.set("spark.sql.execution.arrow.enabled", "true")
            schema = StructType([
            StructField('id_momento', StringType(), False),
            StructField('proyecto', StringType(), False),
            StructField('modelo', StringType(), False),
            StructField('id_indicador', IntegerType(), False),
            StructField('nombre_indicador', StringType(), False),
            StructField('ejecucion', IntegerType(), False),
            StructField('valor_indicador', FloatType(), True)])
            vMetricTable = spark.createDataFrame(vMetricTable, schema=schema)
            vMetricTable.write.format("hive").option("fileFormat","parquet").saveAsTable(metricTable, mode='append')
        
        
             
    def fn_prepare_kcluster(self, queryInputable, categoricalColumns, identificationColumn = None):
        reload(sys)
        sys.setdefaultencoding('utf-8') 
        tranSpark = SparkTransformationTools()
        spark = tranSpark.get_spark_session('anomaly_detector')
        spark.sparkContext.setLogLevel("WARN")
        vDfbegin = spark.sql(queryInputable)
        vDf = vDfbegin.na.drop()
        vInference = tranSpark.fn_get_inferenceData(vDf, categoricalColumns, identificationColumn)
        vInference = vInference.withColumn('Features', col('features'))
        vScaler = StandardScaler(inputCol="Features", outputCol="scaledfeatures",
                                withStd=True, withMean=True)
        vScalerModel = vScaler.fit(vInference)
        vInference = vScalerModel.transform(vInference)
        vInference = vInference.drop('Features')
        vPca = PCA(k=4, inputCol="scaledfeatures", outputCol="features")
        vPcaModel = vPca.fit(vInference)
        vInference = vPcaModel.transform(vInference).select("features")
        vKcluster = self.fn_get_best_clusters(vInference)
        return vKcluster

    def fn_predict(self, queryInputable,  identificationColumn = None,
                   pipelinePath = 'anomalyPipeline',threshold=0.7, 
                   exitTable = 'proceso_enmascarado.frc_resultado_anomalia',
                   metricTable='proceso.monitoreo_frcan'):

        try:
            reload(sys)
            sys.setdefaultencoding('utf-8')
            vDate = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") 
            tranSpark = SparkTransformationTools()
            spark = tranSpark.get_spark_session('anomaly_detector')
            spark.sparkContext.setLogLevel("WARN")
            vDfbegin = spark.sql(queryInputable )
            vDf = vDfbegin.na.drop()
            vModel = PipelineModel.load(pipelinePath)
            vPrediction = vModel.transform(vDf).select(identificationColumn,'prediction', 'probability')
            vFinal = self.fn_get_anomalies(identificationColumn=identificationColumn, threshold=threshold, Prediction=vPrediction)
            vFinal = vFinal.withColumn("probability",col("probability").cast(StringType()))\
            .withColumn("Anomalies", col("Anomalies").cast(IntegerType())).withColumn(identificationColumn,col(identificationColumn).cast(StringType()))
            vFinal.write.format("hive").option("fileFormat","parquet").saveAsTable(exitTable, mode='overwrite')
            vPercetage = vFinal.select('anomalies')
            vTotal  = vPercetage.count()
            vPercetage = vPercetage.groupBy('anomalies')\
            .count().withColumnRenamed('count', 'cnt_por_clase')\
            .withColumn('valor_indicador', (col('cnt_por_clase') / vTotal) * 100 ).sort(col('valor_indicador').desc())
            vPercetage = vPercetage.drop('cnt_por_clase')
            vPandas = vPercetage.toPandas()
            vMetricTable = self.fn_create_metrics_table(vPandas=vPandas, date=vDate, train=False)
            spark.conf.set("spark.sql.execution.arrow.enabled", "true")
            schema = StructType([
            StructField('valor_indicador', FloatType(), True),
            StructField('id_momento', StringType(), False),
            StructField('proyecto', StringType(), False),
            StructField('modelo', StringType(), False),
            StructField('ejecucion', IntegerType(), False),
            StructField('id_indicador', IntegerType(), False),
            StructField('nombre_indicador', StringType(), False),])
            vMetricTable = spark.createDataFrame(vMetricTable, schema=schema)
            vMetricTable.write.format("hive").option("fileFormat","parquet").saveAsTable(metricTable, mode='append')
            vFinal = vFinal.withColumn("probability",col("probability").cast(StringType()))\
            .withColumn("anomalies", col("anomalies").cast(IntegerType())).withColumn(identificationColumn,col(identificationColumn).cast(StringType()))
            vFinal.write.format("hive").option("fileFormat","parquet").saveAsTable(exitTable, mode='overwrite')
            spark.catalog.refreshTable(exitTable)

        
        except Exception as vError:
            print('Error Entrenamiento: {}'.format(vError))
            vMetricTable = self.fn_create_metrics_table(error=True, date=vDate, train=False)
            spark.conf.set("spark.sql.execution.arrow.enabled", "true")
            schema = StructType([
            StructField('id_momento', StringType(), False),
            StructField('proyecto', StringType(), False),
            StructField('modelo', StringType(), False),
            StructField('id_indicador', IntegerType(), False),
            StructField('nombre_indicador', StringType(), False),
            StructField('ejecucion', IntegerType(), False),
            StructField('valor_indicador', FloatType(), True)])
            vMetricTable = spark.createDataFrame(vMetricTable, schema=schema)
            vMetricTable.write.format("hive").option("fileFormat","parquet").saveAsTable(metricTable, mode='append')    
            

    def fn_run_boostrap(self, queryInputable, categoricalColumns, identificationColumn, metricTable = 'proceso.monitoreo_frcan', threshold=0.7, NumCluster = 0):
        
        reload(sys)
        sys.setdefaultencoding('utf-8') 
        tranSpark = SparkTransformationTools()
        spark = tranSpark.get_spark_session('anomaly_detector')
        spark.sparkContext.setLogLevel("WARN")
        vDate = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        vDfbegin = spark.sql(queryInputable)
        vDf = vDfbegin.na.drop().sample(True, 0.5)
        vInference = tranSpark.fn_get_inferenceData(vDf, categoricalColumns, identificationColumn)
        vInference = vInference.withColumn('Features', col('features'))
        vScaler = StandardScaler(inputCol="Features", outputCol="scaledfeatures",
                                withStd=True, withMean=True)
        vScalerModel = vScaler.fit(vInference)
        vInference = vScalerModel.transform(vInference)
        vInference = vInference.drop('Features')
        vPca = PCA(k=4, inputCol="scaledfeatures", outputCol="features")
        vPcaModel = vPca.fit(vInference)
        vInference = vPcaModel.transform(vInference).select("features", identificationColumn)
        vFinal = self.fn_get_anomalies(dataFrame=vInference, identificationColumn=identificationColumn, threshold=threshold, NumCluster=NumCluster)
        vPercetage = vFinal.select('anomalies')
        vTotal  = vPercetage.count()
        vPercetage = vPercetage.groupBy('anomalies').count().withColumnRenamed('count', 'cnt_por_clase').withColumn('valor_indicador', (col('cnt_por_clase') / vTotal) * 100 ).sort(col('valor_indicador').desc())
        vPercetage = vPercetage.drop('cnt_por_clase')
        vPandas = vPercetage.toPandas()
        vMetricTable = self.fn_create_metrics_table(vPandas=vPandas, date=vDate)
        vSilhouetteMetrics = [self.Silhouette, vDate, 'FRICCIONES', 'anomalias_fricciones', 1, 1, 'silhouette_score']
        vClustersMetrics = [self.optimalK, vDate, 'FRICCIONES', 'anomalias_fricciones', 1, 1, 'numero_clusters']
        vLenTable = len(vMetricTable)
        vMetricTable.loc[vLenTable] = vSilhouetteMetrics
        vMetricTable.loc[vLenTable + 1 ] = vClustersMetrics
        spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        vMetricTable = spark.createDataFrame(vMetricTable)
        vMetricTable.write.format("hive").option("fileFormat","parquet").saveAsTable(metricTable, mode='append')
