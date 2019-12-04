BASE_PATH=$1
OUTPUT_PATH=$2

# Find files whose size is between 13M-18M and concatenate them into a single file.
sed -rn '/^[^,]+,O/p' `find $BASE_PATH -name 'part-*.csv' -size +13M -size -18M` > $OUTPUT_PATH