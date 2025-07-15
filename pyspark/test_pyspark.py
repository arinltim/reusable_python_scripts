import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType


python_executable = sys.executable
os.environ['PYSPARK_PYTHON'] = python_executable
os.environ['PYSPARK_DRIVER_PYTHON'] = python_executable

def main():
    """
    This script demonstrates basic DataFrame operations in PySpark including:
    1. Creating a SparkSession.
    2. Creating a DataFrame from sample data.
    3. Writing the DataFrame to disk as CSV and Parquet files.
    4. Reading the data back from the CSV and Parquet files into new DataFrames.
    5. Displaying the contents of the DataFrames.
    """
    # 1. Initialize a SparkSession
    # The SparkSession is the entry point to any Spark functionality.
    # .master("local[*]") runs Spark locally using all available cores.
    # .appName() gives your application a name.
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("PySpark I/O Example") \
        .getOrCreate()

    print("SparkSession created successfully.")

    # 2. Prepare Sample Data and Schema
    # Define the structure of our DataFrame using a schema. This is a best practice
    # as it avoids the overhead of Spark having to infer the schema itself.
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("score", DoubleType(), True)
    ])

    # Create some sample data rows that conform to the defined schema.
    data = [
        (1, "Alice", 34, 85.5),
        (2, "Bob", 45, 90.1),
        (3, "Charlie", 28, 77.3),
        (4, "David", 52, 65.8)
    ]

    # Create the initial DataFrame from the data and schema
    original_df = spark.createDataFrame(data=data, schema=schema)

    print("\n--- Original DataFrame ---")
    original_df.show()
    original_df.printSchema()

    # Define output paths
    csv_path = "spark_output/csv_data"
    parquet_path = "spark_output/parquet_data"

    # 3. Write DataFrame to Disk
    # We will write the DataFrame in two common formats: CSV and Parquet.
    # .mode("overwrite") will replace the directory and its contents if it already exists.

    # Write as CSV
    try:
        print(f"\nWriting DataFrame to CSV at: {csv_path}")
        original_df.write.mode("overwrite").option("header", "true").csv(csv_path)
        print("Successfully wrote to CSV.")
    except Exception as e:
        print(f"An error occurred while writing CSV: {e}")

    # Write as Parquet
    # Parquet is a columnar storage format, highly optimized for big data processing.
    try:
        print(f"\nWriting DataFrame to Parquet at: {parquet_path}")
        original_df.write.mode("overwrite").parquet(parquet_path)
        print("Successfully wrote to Parquet.")
    except Exception as e:
        print(f"An error occurred while writing Parquet: {e}")

    # 4. Read DataFrames from Disk
    # Now, we'll read the data we just wrote back into new DataFrames to verify the process.

    # Read from CSV
    try:
        print(f"\nReading DataFrame from CSV at: {csv_path}")
        # When reading CSV, it's good practice to specify that the file has a header
        # and to infer the schema to get the correct data types.
        csv_df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)

        print("\n--- DataFrame Read from CSV ---")
        csv_df.show()
        csv_df.printSchema()
    except Exception as e:
        print(f"An error occurred while reading CSV: {e}")


    # Read from Parquet
    try:
        print(f"\nReading DataFrame from Parquet at: {parquet_path}")
        # Parquet stores the schema along with the data, so we don't need to infer it.
        # This makes reading from Parquet faster and more reliable.
        parquet_df = spark.read.parquet(parquet_path)

        print("\n--- DataFrame Read from Parquet ---")
        parquet_df.show()
        parquet_df.printSchema()
    except Exception as e:
        print(f"An error occurred while reading Parquet: {e}")


    # 5. Stop the SparkSession
    # It's important to stop the SparkSession to release the resources.
    spark.stop()
    print("\nSparkSession stopped.")

if __name__ == '__main__':
    main()
