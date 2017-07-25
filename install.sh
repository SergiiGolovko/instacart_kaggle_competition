# 1. Clone the project.
git clone https://github.com/SergiiGolovko/instacart_kaggle_competition
# 2. cd to root.
cd instacart_kaggle_competition
# 3. Create the environment.
conda env create -f requirements.yml
# 4. Copy data from s3 bucket.
aws s3 cp s3://sergii-golovko-instacart ./data/ --recursive
cd data
unzip aisles.csv.zip
unzip departments.csv.zip
unzip order_products__prior.csv.zip
unzip order_products__train.csv.zip
unzip orders.csv.zip
unzip products.csv.zip
