from anomalydetector import AnomalyDetector
from kmeans import KmeansModel
from tranformariontools import SparkTransformationTools
from six.moves.configparser import ConfigParser 
import json 
import sys
import os


def main():
      reload(sys)
      path = os.path.abspath(os.path.dirname(__file__))
      path_config = os.path.join(path, 'config.json')
      with open(path_config) as infile:
            tabla = json.load(infile)["parametros_modelo"]
      exit_table_anomalies = tabla["tablas_salida"]["exitTable_anomalies_predict"]
      identificationColumn_anomalies = tabla["columnas_identificacion"]["anomalias"]
      predict_input_anomailes = tabla["consultas"]["anomalias_prediccion"]
      threshold = float(tabla["threshold"])
      pipeline_path_anomalies = tabla["modelos"]["anomalias"]
      metric_table_anomalies = tabla["tablas_monitoreo"]["monitoring_anomalies"]
      exit_table_segmentation = tabla["tablas_salida"]["exitTableSegmentation"]
      predict_input_segmentation = tabla["consultas"]["segmentacion_prediccion"]
      identificationColumn_segmentation = tabla["columnas_identificacion"]["segmentacion"]
      pipeline_path_segmentation = tabla["modelos"]["segmentacion"]
      metric_table_segmentation = tabla["tablas_monitoreo"]["monitoring_segmentation"]
      result_prediction_query= tabla["consultas"]["cruce_resultado_prediccion"]
      result_enriched_predict = tabla["tablas_salida"]["exitTable_resultado_enriquecido_predict"]
      tranSpark = SparkTransformationTools()
      spark = tranSpark.get_spark_session('anomaly_detector')
      spark.sparkContext.setLogLevel("ERROR")
      
      vDetector = AnomalyDetector()
      vDetector.fn_predict(queryInputable=predict_input_anomailes,
                             identificationColumn=identificationColumn_anomalies,
                             exitTable=exit_table_anomalies,
                             threshold = threshold,
                             pipelinePath= pipeline_path_anomalies,
                             metricTable = metric_table_anomalies)
      
       
       
      vTemporalresult = spark.sql(result_prediction_query).dropDuplicates([identificationColumn_segmentation])
      vTemporalresult.write.format("hive").option("fileFormat","parquet").mode("overwrite").saveAsTable(result_enriched_predict)
      spark.catalog.refreshTable(result_enriched_predict)


      vSegmenter = KmeansModel()
      vSegmenter.fn_predict(queryInputable=predict_input_segmentation,
                            identificationColumn=identificationColumn_segmentation,
                            pipelinePath=pipeline_path_segmentation,
                            exitTable=exit_table_segmentation,
                            metricTable=metric_table_segmentation)
      
if __name__ == '__main__':
      main()
    
    
    


