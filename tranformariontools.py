from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col


class SparkTransformationTools:
    def __init__(self):
        self.spark = None
   
    def get_spark_session(self, name):
        spark = SparkSession \
        .builder \
        .appName(name) \
        .config("spark.sql.warehouse.dir", '/warehouse/tablespace/external/hive')\
        .config("hive.exec.dynamic.partition.mode", "nonstrict")\
        .config("hive.exec.dynamic.partition", "true")\
        .enableHiveSupport().getOrCreate()
	
        # spark.conf.set("spark.network.timeout", "86400s")
        return spark

    def load_external_data(self, format, path, 
                       spark,schema):
        df_data = (spark.read.  
        format(format). \
        schema(schema).  
        #option('inferSchema', 'True').
        #option('sep', ';').
        option('header', 'True'). \
        load(path))
        return df_data 

    def fn_vectorize_columns(self,
                             listVariables,
                             outputCol =  'Features'):
        
        vVectorizer =  VectorAssembler(outputCol= outputCol)
        vVectorizer.setInputCols(listVariables)

        return vVectorizer


    def fn_get_train_test_data(self, dataFrame,
                               targetVariable,
                               listVariables,
                               trainSize = 0.7,
                               returnTrain = True):
        vVariablesvector = VectorAssembler(inputCols=listVariables,
                                           outputCol='X')
        vVariablesvector = vVariablesvector.transform(dataFrame)
        vModeldata = vVariablesvector.select('X', targetVariable)
        vTrain, vTest = vModeldata.randomSplit([trainSize, 1 - trainSize])
        if returnTrain is True:
            return vTrain
        else:
            return vTest


    def fn_create_indexers_dictionary(self,
                                      categoricalColumns):
        self.vListOutputCol = []
        self.vDictIndexers = []
        for variable in categoricalColumns:
            vOuputcol = variable +'label'
            self.vListOutputCol.append(vOuputcol)
            self.vDictIndexers.append(StringIndexer(inputCol=variable, outputCol=vOuputcol).setHandleInvalid("keep"))

        return self.vDictIndexers,self.vListOutputCol


    def fn_create_encoder_dictionary(self,
                                     categoricalColumns):
        
        self.vListVectorCol = []  
        self.vDictEncoder = []
        for variable in categoricalColumns:
            vOuputvector = variable +'Vector'
            self.vListVectorCol.append(vOuputvector)
            self.vDictEncoder.append(OneHotEncoder(inputCol=variable, outputCol=vOuputvector))

        return self.vDictEncoder, self.vListVectorCol
    
    
    
    def fn_get_preparationStages(self, categoricalColumns, dataframeColumns, identificationColumn):
        
        stages = []
        if categoricalColumns[0] != "":
            vDictIndexers, vListOutputCol = self.fn_create_indexers_dictionary(categoricalColumns)
            vDictEncoder, vListVectorCol = self.fn_create_encoder_dictionary(vListOutputCol)
            stages.extend(vDictIndexers)
            stages.extend(vDictEncoder)
        else:
            vListVectorCol = []
        vListnumericColumns = [elemt for elemt in list(dataframeColumns) if elemt not in categoricalColumns]
        vListFinalColumns = vListnumericColumns + vListVectorCol
        if identificationColumn in vListFinalColumns:
            vListFinalColumns.remove(identificationColumn)
        vVectorizer = self.fn_vectorize_columns(vListFinalColumns, outputCol = "Features")
        stages.append(vVectorizer)
        return stages
        
    def fn_prepare_data(self, dataFrame,
                        categoricalColumns,
                        targetVariable,
                        listVariables = None,
                        trainSize = 0.7,
                        returnTrain = True):
        
        
        vListFinalColumns = list(dataFrame.columns)
        for element in categoricalColumns:
            vListFinalColumns.remove(element)
        vListFinalColumns.remove(targetVariable)
        dataFrame = self.fn_create_indexers_dictionary(categoricalColumns)
        dataFrame = self.fn_create_encoder_dictionary(self.vListOutputCol)
        vListFinalColumns += self.vListVectorCol
	print("vlistcolums"+str(vListFinalColumns))
        vTrain = self.fn_get_train_test_data(dataFrame, targetVariable=targetVariable,
                                        listVariables=vListFinalColumns, trainSize=trainSize,  returnTrain=True)
        vTest = self.fn_get_train_test_data(dataFrame, targetVariable=targetVariable,
                                        listVariables=vListFinalColumns, trainSize=trainSize,  returnTrain=False)
        return vTrain, vTest
        
    def fn_get_inferenceData(self, dataFrame,
                             categoricalColumns,
                             identificationColumn = None):
            vListnumericColumns = [elemt for elemt in list(dataFrame.columns) if elemt not in categoricalColumns]
            vdataFrame = self.fn_create_indexers_dictionary(dataFrame, categoricalColumns)
            vdataFrame = self.fn_create_encoder_dictionary(vdataFrame, self.vListOutputCol)            
            vListFinalColumns = vListnumericColumns + self.vListVectorCol
            if identificationColumn is not None:
            	vListFinalColumns.remove(identificationColumn)
            vdataFrame = self.fn_vectorize_columns(vdataFrame, vListFinalColumns)
            if identificationColumn is None:
                return vdataFrame.select('features')
            else:
                return  vdataFrame.select(identificationColumn, 'features')    
    
    def fn_save_table(self, dataframe, formats, spark, table_name):
        dataframe.registerTempTable("temp_table")
        spark.sql("DROP TABLE IF EXISTS {} PURGE".format(table_name))
        spark.sql("CREATE TABLE IF NOT EXISTS {} STORED AS parquet AS SELECT * FROM temp_table".format(table_name))
    
