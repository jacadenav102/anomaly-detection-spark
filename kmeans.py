
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql.functions import row_number
from pyspark.ml import Pipeline, PipelineModel 
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler
from tranformariontools import SparkTransformationTools
from pyspark.sql.functions import col
from pyspark.sql.types import *
import sys
import numpy as np
import pandas as pd
import datetime

class KmeansModel:
    def __init__(self):
        pass


    def fn_create_metrics_table(self, date, vPandas = None ,error = False, train=True):
        
        
        
        if train is True:
            vType = 'segmentacion_anomalias_fricciones_ent'
        
        else:
            vType = 'segmentacion_anomalias_fricciones_pre'
        
        if error is False:
            vPandas = vPandas.sort_values('valor_indicador')
            vPandas['id_momento'] = date
            vPandas['proyecto'] = 'FRICCIONES'
            vPandas['modelo'] = vType
            vPandas['ejecucion'] = 1
            vPandas['id_indicador'] = 2 
            vListlabels = []
            for i in range(len(vPandas['valor_indicador'])):
                vListlabels.append('anomalias_gf_' +  str(i))
            vPandas['nombre_indicador'] = vListlabels
            vPandas = vPandas.drop('prediction', axis = 1)
            
        else:
            vListMonitoring = []
            vListMonitoring.append(datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
            vListMonitoring.append('FRICCIONES')
            vListMonitoring.append(vType)
            vListMonitoring.append(2)
            vListMonitoring.append('anomalias_gf_0')
            vListMonitoring.append(0)
            vListMonitoring.append(None)

            vPandas = pd.DataFrame()
            vPandas = vPandas.append([vListMonitoring])
            vPandas.columns = ['id_momento', 'proyecto', 'modelo', 'id_indicador',
                                'nombre_indicador', 'ejecucion', 'valor_indicador'] 
                
        return vPandas


    def fn_get_best_clusters(self,dataFrame,
                             maximunK = 6,
                             seed = 123):
        vListSilhouette = []
        vListKcluster = []
        evaluator = ClusteringEvaluator()  
        for vK in range(3, maximunK + 1):
            try: 
                vKmeans = KMeans(k=vK, seed=seed)
                model = vKmeans.fit(dataFrame)
                vPrediction = model.transform(dataFrame)
                vSilhouette = evaluator.evaluate(vPrediction)
                vListSilhouette.append(vSilhouette)
                vListKcluster.append(vK)
            except:
                continue 
            
        if  not vListSilhouette:
            return 3
        else:
            vIndex = np.argmax(vListSilhouette)
            return vListKcluster[vIndex]

    def fn_get_clusters(self, dataFrame,
                       seed=123,
                       maximunK = 6,
                       NumCluster = 0,
                       trainedModel = None):
        
        if trainedModel is not None:
            
            vPrediction = trainedModel.transform(dataFrame)
            return vPrediction
        
        else:
        
            if NumCluster == 0:
                vNumCluster = self.fn_get_best_clusters(dataFrame=dataFrame,
                                            maximunK=maximunK,
                                            seed=seed)

            else:
                vNumCluster = NumCluster
            
            vKms= KMeans(k=vNumCluster, seed=seed)
            self.model = vKms.fit(dataFrame)
            vPrediction =self.model.transform(dataFrame)
            return vPrediction
    
    def fn_save_model(self, path):
        self.model.write().overwrite().save(path)


    def fn_train_model(self, spark ,queryInputable, categoricalColumns, identificationColumn, metricTable = 'proceso_enmascarado.monitoreo_frcse',
                    pipelinePath = 'kmeansPipeline', seed = 123):
        try:
            reload(sys)
            sys.setdefaultencoding('utf-8')
            vDate = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") 
            tranSpark = SparkTransformationTools()
            spark.sparkContext.setLogLevel("WARN")
            vDfbegin = spark.sql(queryInputable)
            vDf = vDfbegin.na.drop()
            stages = tranSpark.fn_get_preparationStages(categoricalColumns, vDf.columns, identificationColumn)
            vScaler = StandardScaler(inputCol="Features", outputCol="features",
                                    withStd=True, withMean=True)
            stages.append(vScaler)
            vPrePipeline = Pipeline(stages=stages).fit(vDf)
            vInference = vPrePipeline.transform(vDf).select("features", identificationColumn)
            vNumCluster = self.fn_get_best_clusters(vInference, maximunK=4)
            finalStages = vPrePipeline.stages
            vKms= KMeans(k=vNumCluster, seed=seed).fit(vInference )
            vFinal = vKms.transform(vInference)
            finalStages.append(vKms)
            trainedPL = PipelineModel(finalStages)
            trainedPL.write().overwrite().save(pipelinePath)
            vTotal  = vFinal.count()
            vPercetage = vFinal.groupBy('prediction').count().withColumnRenamed('count', 'cnt_por_cluster').withColumn('valor_indicador', (col('cnt_por_cluster') / vTotal) * 100 ).sort(col('valor_indicador').desc())
            vPercetage = vPercetage.drop('cnt_por_cluster')
            vPandas = vPercetage.toPandas()
            vMetricTable = self.fn_create_metrics_table(vPandas=vPandas, date=vDate, train= True)
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

           
        except Exception as vError:
            print('Error Entrenamiento: {}'.format(vError))
            vMetricTable = self.fn_create_metrics_table(error=True, date=vDate, train= True)
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
         

    def fn_load_models(self, queryInputable, categoricalColumns,
                       identificationColumn, scalerPath = 'kmeansScaler',
                       modelPath = 'kmeans'):
        tranSpark = SparkTransformationTools()
        spark = tranSpark.get_spark_session('segmentation_process')
        vDfbegin = spark.sql(queryInputable + ' LIMIT 50')
        vDf = vDfbegin.na.drop()
        vInference = tranSpark.fn_get_inferenceData(vDf, categoricalColumns, identificationColumn)
        vInference = vInference.withColumn('Features', col('features'))
        vScaler = StandardScaler(inputCol="Features", outputCol="features", withStd=True, withMean=False)
        vScalerModel = vScaler.fit(vInference)
        vInference = vScalerModel.transform(vInference)
        vFinal = self.fn_get_clusters(vInference, 3)
        vModel = self.model
        vModel = vModel.load(modelPath)
        vScalerModel = vScalerModel.load(scalerPath)
        return (vModel, vScalerModel)
        
    def fn_predict(self,queryInputable, 
                   identificationColumn,
                   exitTable='proceso_enmascarado.frc_resultado_segmentacion',
                   pipelinePath = 'kmeansPipeline',
                   metricTable='proceso.monitoreo_frcse'):

        try:
            queryInputable = queryInputable 
            reload(sys)
            sys.setdefaultencoding('utf-8')
            vDate = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") 
            tranSpark = SparkTransformationTools()
            spark = tranSpark.get_spark_session('segmentation_process')
            spark.sparkContext.setLogLevel("WARN")
            vDfbegin = spark.sql(queryInputable)
            vDf = vDfbegin.na.drop()
            vModel = PipelineModel.load(pipelinePath)
            vFinal = vModel.transform(vDf )
            vFinal.select("prediction", identificationColumn)
            vFinal = vFinal.withColumn(identificationColumn,col(identificationColumn).cast(StringType())).withColumn("prediction", col("prediction").cast(IntegerType()))
            vFinal.write.format("hive").option("fileFormat","parquet").saveAsTable(exitTable, mode='overwrite')
            spark.catalog.refreshTable(exitTable)
            vTotal  = vFinal.count()
            vPercetage = vFinal.groupBy('prediction').count().withColumnRenamed('count', 'cnt_por_cluster').withColumn('valor_indicador', (col('cnt_por_cluster') / vTotal) * 100 ).sort(col('valor_indicador').desc())
            vPercetage = vPercetage.drop('cnt_por_cluster')
            vPandas = vPercetage.toPandas()
            vMetricTable = self.fn_create_metrics_table(vPandas=vPandas, date=vDate, train= False)
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
        
        except Exception as vError:
            print('Error Prediccion: {}'.format(vError))
            vMetricTable = self.fn_create_metrics_table(error=True, date=vDate, train= False)
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
        
        
    def fn_get_prediction(self, dataFrame,
                        identificationColumn = None,
                        maximunK = 6,
                        seed = 123,
                        NumCluster = 0):
        vPrediction = self.fn_get_clusters(dataFrame=dataFrame,
                                        maximunK=maximunK,
                                        seed=seed,
                                        NumCluster=NumCluster)
        if identificationColumn is None:
            return vPrediction.select('probability', fn_check_probabilities('probability').alias('Anomalies'))

        else:
            prediction = vPrediction.select('probability', fn_check_probabilities('probability').alias('Anomalies')).withColumn('row_index', row_number().over(Window.orderBy(monotonically_increasing_id())))
            infereceData = dataFrame.select(identificationColumn).withColumn('row_index', row_number().over(
            vDataresult = infereceData.join(prediction, on=["row_index"]).drop("row_index")))
            return vDataresult

    def fn_prepare_kcluster(self,queryInputable, categoricalColumns, identificationColumn):
        reload(sys)
        sys.setdefaultencoding('utf-8') 
        tranSpark = SparkTransformationTools()
        spark = tranSpark.get_spark_session('segmentation_process')
        spark.sparkContext.setLogLevel("WARN")
        vDfbegin = spark.sql(queryInputable)
        vDf = vDfbegin.na.drop()
        vInference = tranSpark.fn_get_inferenceData(vDf, categoricalColumns, identificationColumn)
        vInference = vInference.withColumn('Features', col('features'))
        vScaler = StandardScaler(inputCol="Features", outputCol="features",
                                withStd=True, withMean=False)
        vScalerModel = vScaler.fit(vInference)
        vInference = vScalerModel.transform(vInference)
        vInference = vInference.select("features")
        vKcluster = self.fn_get_best_clusters(vInference)
        return vKcluster

    def fn_run_boostrap(self, queryInputable, categoricalColumns, identificationColumn, metricTable, NumCluster=0):
        reload(sys)
        sys.setdefaultencoding('utf-8') 
        vDate = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S") 
        tranSpark = SparkTransformationTools()
        spark = tranSpark.get_spark_session('segmentation_process')
        spark.sparkContext.setLogLevel("WARN")
        vDfbegin = spark.sql(queryInputable).sample(True, 1.0)
        vDf = vDfbegin.na.drop()
        vInference = tranSpark.fn_get_inferenceData(vDf, categoricalColumns, identificationColumn)
        vInference = vInference.withColumn('Features', col('features'))
        vScaler = StandardScaler(inputCol="Features", outputCol="features", withStd=True, withMean=False)
        vScalerModel = vScaler.fit(vInference)
        vInference = vScalerModel.transform(vInference)
        vFinal = self.fn_get_clusters(vInference, NumCluster=NumCluster)
        vTotal  = vFinal.count()
        vPercetage = vFinal.groupBy('prediction').count().withColumnRenamed('count', 'cnt_por_cluster').withColumn('valor_indicador', (col('cnt_por_cluster') / vTotal) * 100 ).sort(col('valor_indicador').desc())
        vPercetage = vPercetage.drop('cnt_por_cluster')
        vPandas = vPercetage.toPandas()
        vMetricTable = self.fn_create_metrics_table(vPandas=vPandas, date=vDate)
        vMetricTable = spark.createDataFrame(vMetricTable)
        vMetricTable.write.format("hive").option("fileFormat","parquet").saveAsTable(metricTable, mode='append')

