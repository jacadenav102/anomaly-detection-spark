from anomalydetector import AnomalyDetector
from kmeans import KmeansModel
from tranformariontools import SparkTransformationTools
from six.moves.configparser import ConfigParser 
import sys
import json
import os

def main():
    reload(sys)
    path = os.path.abspath(os.path.dirname(__file__))
    path_config = os.path.join(path, 'config.json')
    with open(path_config) as infile:
        tabla = json.load(infile)["parametros_modelo"]
    exit_table_anomalies = tabla["tablas_salida"]["exitTable_anomalies_train"]
    exit_table_segmentation = tabla["tablas_salida"]["exitTableSegmentation"]
    identificationColumnAnomalies = tabla["columnas_identificacion"]["anomalias"]
    identificationColumnSegmentation = tabla["columnas_identificacion"]["segmentacion"]
    train_input_anomalies = tabla["consultas"]["anomalias_entrenamiento"]
    train_input_segmentation = tabla["consultas"]["segmentacion_entrenamiento"]
    threshold = float(tabla["threshold"])
    pipeline_path_anomalies = tabla["modelos"]["anomalias"]
    pipeline_path_segmentation = tabla["modelos"]["segmentacion"]
    metric_table_anomalies = tabla["tablas_monitoreo"]["monitoring_anomalies"]
    metric_table_segmentation = tabla["tablas_monitoreo"]["monitoring_segmentation"]
    categorical_columns_anomalies = tabla["columnas_categoricas"]["anomalias"].split(",")
    categorical_columns_segmentation = tabla["columnas_categoricas"]["segmentacion"].split(",")
    result_train_query= tabla["consultas"]["cruce_resultado_entrenamiento"]
    result_enriched_train = tabla["tablas_salida"]["exitTable_resultado_enriquecido_train"]
    tranSpark = SparkTransformationTools()
    spark = tranSpark.get_spark_session('anomaly_detector')
    spark.sparkContext.setLogLevel("WARN")


    
    vDetector = AnomalyDetector()
    vDetector.fn_train_model(queryInputable=train_input_anomalies, 
                             categoricalColumns=categorical_columns_anomalies,
                             identificationColumn=identificationColumnAnomalies,
                             exitTable=exit_table_anomalies,
                             threshold = threshold,
                             metricTable = metric_table_anomalies,
                             pipelinePath= pipeline_path_anomalies)
    
    
    
    vTemporalresult =  spark.sql(result_train_query).dropDuplicates([identificationColumnSegmentation])    
    vTemporalresult.write.format("hive").option("fileFormat","parquet").mode("overwrite").saveAsTable(result_enriched_train)
    spark.catalog.refreshTable(result_enriched_train)
    
    vSegmenter = KmeansModel()
    vSegmenter.fn_train_model(queryInputable=train_input_segmentation, 
                            categoricalColumns=categorical_columns_segmentation,
                            identificationColumn=identificationColumnSegmentation,
                            pipelinePath=pipeline_path_segmentation,
                            metricTable=metric_table_segmentation)

if __name__ == '__main__':
	main()