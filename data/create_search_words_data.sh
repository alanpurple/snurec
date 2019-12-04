BASE_PATH=$1
OUTPUT_PATH=$2

# Extract 4th column from user action data. The 4th column refers to search keyword.
# If there was no search action, the 4th column will be "".
cat ${BASE_PATH}/part-*.csv | grep -oP "^[^,]+,[^,]+,[^,]+,\K[^,]+" > ${OUTPUT_PATH}

# Remove lines with ""
sed '/^""$/d' ${OUTPUT_PATH} > ${OUTPUT_PATH}