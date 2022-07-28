SOURCE_BASE_PATH="/collaborative_filtering"

INPUT_HADOOP_DIR="/collaborative_filtering/input"
OUTPUT_HADOOP_DIR="/collaborative_filtering/output"

HADOOP_STREAMING_PATH="${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar"

hdfs dfs -test -d ${INPUT_HADOOP_DIR}
if [ $? -eq 0 ];
  then
    echo "Remove ${INPUT_HADOOP_DIR}"
    hdfs dfs -rm -r ${INPUT_HADOOP_DIR}
fi

hdfs dfs -test -d ${OUTPUT_HADOOP_DIR}
if [ $? -eq 0 ];
  then
    echo "Remove ${OUTPUT_HADOOP_DIR}"
    hdfs dfs -rm -r ${OUTPUT_HADOOP_DIR}
fi

test -d ${SOURCE_BASE_PATH}/data/output
if [ $? -eq 0 ];
  then
    echo "Remove ${SOURCE_BASE_PATH}/data/output"
    rm -rf ${SOURCE_BASE_PATH}/data/output
fi

hdfs dfs -mkdir -p ${INPUT_HADOOP_DIR}
hdfs dfs -copyFromLocal ${SOURCE_BASE_PATH}/data/input/* ${INPUT_HADOOP_DIR}

hdfs dfs -ls /collaborative_filtering/input
chmod 0777 ${SOURCE_BASE_PATH}/src

echo "stage 1"
hadoop jar ${HADOOP_STREAMING_PATH} \
  -D mapreduce.job.maps=8 \
  -D mapreduce.job.reduces=4 \
  -files ${SOURCE_BASE_PATH}/src \
  -mapper src/mapper_1.py \
  -reducer src/reducer_1.py \
  -input ${INPUT_HADOOP_DIR}/ratings.csv \
  -output ${OUTPUT_HADOOP_DIR}/stage_1 \

echo "stage 2"
hadoop jar ${HADOOP_STREAMING_PATH} \
  -D mapreduce.job.maps=8 \
  -D mapreduce.job.reduces=8 \
  -files ${SOURCE_BASE_PATH}/src \
  -mapper src/mapper_2.py \
  -reducer src/reducer_2.py \
  -input ${OUTPUT_HADOOP_DIR}/stage_1 \
  -output ${OUTPUT_HADOOP_DIR}/stage_2 \

echo "stage 3"
hadoop jar ${HADOOP_STREAMING_PATH} \
  -D mapreduce.job.maps=8 \
  -D mapreduce.job.reduces=8 \
  -files ${SOURCE_BASE_PATH}/src \
  -mapper src/mapper_3.py \
  -reducer src/reducer_3.py \
  -input ${OUTPUT_HADOOP_DIR}/stage_2,${INPUT_HADOOP_DIR}/ratings.csv \
  -output ${OUTPUT_HADOOP_DIR}/stage_3 \
 
echo "stage 4"
hadoop jar ${HADOOP_STREAMING_PATH} \
  -D mapreduce.job.maps=8 \
  -D mapreduce.job.reduces=8 \
  -D mapreduce.map.memory.mb=128 \
  -D mapreduce.map.java.opts=-Xmx96m \
  -D mapreduce.reduce.memory.mb=128 \
  -D mapreduce.reduce.java.opts=-Xmx96m \
  -files ${SOURCE_BASE_PATH}/src \
  -mapper src/mapper_4.py \
  -reducer src/reducer_4.py \
  -input ${OUTPUT_HADOOP_DIR}/stage_3,${INPUT_HADOOP_DIR}/ratings.csv \
  -output ${OUTPUT_HADOOP_DIR}/stage_4 \

echo "stage 5"
hadoop jar ${HADOOP_STREAMING_PATH} \
  -D mapreduce.job.output.key.comparator.class=org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator \
  -D stream.num.map.output.key.fields=3 \
  -D mapreduce.partition.keycomparator.options="-k1,1n -k2,3nr -k3,2" \
  -D mapreduce.partition.keypartitioner.options=-k1,1n \
  -files ${SOURCE_BASE_PATH}/src,${SOURCE_BASE_PATH}/data/input/movies.csv \
  -mapper src/mapper_5.py \
  -reducer src/reducer_5.py \
  -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
  -input ${OUTPUT_HADOOP_DIR}/stage_4 \
  -output ${OUTPUT_HADOOP_DIR}/final \

hdfs dfs -copyToLocal ${OUTPUT_HADOOP_DIR} ${SOURCE_BASE_PATH}/data

hdfs dfs -rm -r ${INPUT_HADOOP_DIR}
hdfs dfs -rm -r ${OUTPUT_HADOOP_DIR}
