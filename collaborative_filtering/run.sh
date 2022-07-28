OUTPUT_DIR="data/output"

docker cp ../collaborative_filtering namenode:/

exec_args="sh /collaborative_filtering/run_hadoop.sh"
docker exec -it namenode sh -c "${exec_args}"

if [ "$(ls -A $OUTPUT_DIR)" ]; then
     echo "clearing $OUTPUT_DIR"
     rm -rf $OUTPUT_DIR/*
else
    echo "$OUTPUT_DIR is Empty"
fi

docker cp namenode:/collaborative_filtering/data/output data


