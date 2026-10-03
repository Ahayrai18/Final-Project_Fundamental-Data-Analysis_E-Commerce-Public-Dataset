# Brazilian E-Commerce Data Analysis 🛒

An exploratory data analysis project using the **Brazilian E-Commerce Public Dataset by Olist**. The project looks at marketplace transactions in Brazil from **2016 to 2018**, with a focus on product categories, sales trends, customer locations, delivery performance, customer reviews, customer value, and seller–customer distance.

The project includes a fully executed Jupyter Notebook for data cleaning and analysis, plus an interactive Streamlit dashboard for exploring the results.

> **Note:** This is a descriptive analysis project. It uses aggregation, descriptive statistics, visualization, correlation, RFM analysis, geospatial analysis, and manual grouping/binning. No machine learning model is used.

## Business Questions

The analysis was built around three practical questions:

1. **Product categories**  
   Which product categories generate the highest and lowest revenue (top five and bottom five) among *delivered* orders from January 2017 to August 2018, and what percentage of total revenue comes from the top five categories?  
   *Action:* understand which categories contribute most to marketplace revenue and provide a basis for inventory, promotion, and seller-acquisition planning.

2. **Sales trend**  
   How do monthly order volume and revenue change from January 2017 to August 2018, and what is the year-over-year growth for January–August 2018 compared with January–August 2017?  
   *Action:* understand marketplace growth and support planning for logistics capacity and campaign timing.

3. **Customer location and delivery**  
   Which Brazilian state has the largest customer base, which state has the longest average delivery time and the lowest review score, and how different are review scores between late and on-time orders during January 2017 to August 2018?  
   *Action:* identify geographic areas where delivery performance and customer experience can be examined more closely.

### Additional analysis

The project also includes three additional analyses:

- RFM customer segmentation
- Geospatial analysis of customer and seller locations
- Manual grouping/binning for Brazilian regions and seller–customer distance

These analyses are used to turn the transactional data into customer, logistics, and geographic patterns that are easier to interpret.

## Dataset

The project uses the **Brazilian E-Commerce Public Dataset by Olist**, which contains approximately 100,000 marketplace orders together with information about customers, products, sellers, payments, reviews, order status, and Brazilian geolocation data.

### Dataset files

The original dataset contains nine related tables:

| File | Description | Rows |
|---|---|---:|
| `customers_dataset.csv` | Customer IDs, location, city, and state | 99,441 |
| `geolocation_dataset.csv.gz` | Brazilian ZIP-code coordinates | 1,000,163 |
| `order_items_dataset.csv` | Products included in each order, price, freight, and seller | 112,650 |
| `order_payments_dataset.csv` | Payment type, installments, and payment value | 103,886 |
| `order_reviews_dataset.csv` | Review scores, comments, and review timestamps | 99,224 |
| `orders_dataset.csv` | Order status and order timestamps | 99,441 |
| `product_category_name_translation.csv` | Portuguese-to-English category translation | 71 |
| `products_dataset.csv` | Product category, dimensions, weight, and photo information | 32,951 |
| `sellers_dataset.csv` | Seller IDs and location information | 3,095 |

Together, the nine source tables contain **1,550,922 rows**.

### Data source

- [Brazilian E-Commerce Public Dataset by Olist – Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The dataset was published by Olist and contains real but anonymized commercial transaction data from its marketplace. The dataset covers orders made between 2016 and 2018. The Kaggle listing states that the dataset is licensed under **CC BY-NC-SA 4.0**. For reuse, check the original dataset page for the complete license and attribution terms.

## Project Results

The following findings are based on the cleaned and filtered data used in the notebook and dashboard. Unless stated otherwise, the main business analysis uses *delivered* orders with valid delivery dates from **January 2017 through August 2018**.

### 1. Product categories

The five categories with the largest revenue are:

| Rank | Category | Revenue |
|---|---|---:|
| 1 | Health Beauty | R$ 1.23 million |
| 2 | Watches Gifts | R$ 1.16 million |
| 3 | Bed Bath Table | R$ 1.02 million |
| 4 | Sports Leisure | R$ 0.95 million |
| 5 | Computers Accessories | R$ 0.89 million |

Together, these five categories contribute **39.9% of total revenue**, which is R$ 13.18 million for the filtered analysis period.

Only **18 of 73 categories** are needed to reach 80% of cumulative revenue, showing a strong concentration of revenue in a relatively small part of the catalog.

The five categories with the smallest revenue are:

- Security And Services — R$ 283
- Fashion Childrens Clothes — R$ 520
- Cds Dvds Musicals — R$ 730
- Home Comfort 2 — R$ 760
- Flowers — R$ 1,110

Combined, the five smallest categories account for only **0.03% of total revenue**.

Average item prices also differ considerably between categories, so a category with many units sold is not necessarily the category contributing the most revenue.

### 2. Sales trend

The marketplace experienced rapid growth during 2017.

- Orders increased from **750 in January 2017** to a peak of **7,288 in November 2017**.
- November 2017 was the highest monthly order volume in the analysis period and coincided with the Black Friday period.
- During 2018, monthly orders remained in the range of approximately **6,096–7,069 orders**.
- January–August 2018 compared with January–August 2017:
  - Orders increased by **139.9%**
  - Revenue increased by **141.1%**
- Average revenue per order changed only slightly, from **R$ 136.1** to **R$ 136.7**.

The year-over-year percentages are strongly affected by the small starting base in early 2017. For example, January order growth is **+843%**, while the increase falls to about **+51% in August**.

### 3. Customer location and delivery

Customer distribution is highly concentrated geographically.

- **SP** contains **41.9% of customers**.
- The five largest states — **SP, RJ, MG, RS, and PR** — account for **77.1% of customers**.

For states with at least 100 orders, the analysis found:

- Fastest average delivery: **SP — 8.7 days**
- Slowest average delivery: **AM — 26.4 days**
- Lowest review score among the compared states: **MA — 3.83**

Across all analyzed orders:

- **8.1%** of orders were late.
- Late orders received an average review score of **2.57**.
- On-time orders received an average review score of **4.29**.
- The difference is **1.73 review-score points**.

Average review scores also decrease as delivery time increases, from approximately **4.42** for orders delivered within seven days to **3.12** for orders taking more than 21 days.

## Data Cleaning

The notebook follows a structured data-wrangling process before the exploratory analysis.

### 1. Load and inspect the source data

All nine source tables are loaded into pandas DataFrames and checked for:

- missing values
- duplicates
- data types
- inconsistent dates
- invalid values
- inconsistent labels
- time coverage
- outliers

### 2. Validate order-level relationships

The analysis checks the relationships between order, item, payment, review, customer, product, seller, and geolocation tables.

A key point is that `orders` contains **99,441 orders**, while `order_items` contains **112,650 rows**, because a single order can contain multiple products and can involve multiple sellers.

### 3. Handle missing and inconsistent values

The notebook identifies several incomplete fields:

- order approval, carrier handover, and customer delivery timestamps are missing for unfinished orders
- some products do not have a category
- review titles and comments contain many missing values

For the main business analysis, only *delivered* orders with valid delivery timestamps are retained.

Product categories that are still missing are labeled as `unknown`, while review text is not required for the business questions and is therefore not used as an analytical feature.

### 4. Handle duplicate records

`geolocation` contains a large number of repeated ZIP-code observations, so it is reduced to one representative point per ZIP-code prefix.

The review table also contains repeated review IDs and some orders with multiple reviews. The latest review is retained per order for the analysis.

### 5. Validate dates and order status

The notebook checks for inconsistent combinations between order status and timestamps, including:

- delivered orders without a delivery timestamp
- non-delivered orders with a delivery timestamp
- delivery events occurring before carrier handover

Only delivered orders with valid delivery dates are used in the final business-analysis period.

### 6. Handle invalid values

The notebook identifies and removes invalid geolocation coordinates outside Brazil.

Payment fields and product weight are inspected for invalid values, but those fields are not required for the main business questions.

### 7. Normalize labels

Product categories are translated from Portuguese to English using the category translation table. Two categories missing from the translation table are completed manually:

- `pc_gamer` → `pc_gamer`
- `portateis_cozinha_e_preparadores_de_alimentos` → `portable kitchen appliances`

City names are not used as the main geographic ranking variable because their spelling can differ. The analysis uses Brazilian state codes instead.

### 8. Use the correct customer identifier

`customer_id` identifies the customer attached to a specific order, while `customer_unique_id` is used to identify the same person across different orders.

For customer counts and RFM analysis, the project uses:

```text
customer_unique_id
```

This allows repeat customers to be identified correctly.

### 9. Restrict the main analysis period

The source orders span approximately September 2016 to October 2018, but only **January 2017 to August 2018** is used for the main analysis because this period provides complete monthly coverage.

### 10. Keep meaningful outliers

The notebook reviews unusual values such as very high product prices and very long delivery times.

These observations are kept because they can represent genuine transactions rather than data-entry errors. The goal is to describe the marketplace rather than remove every statistically unusual record.

### 11. Create analysis-ready tables

The cleaned data is organized into two main analytical levels:

- `sales`: one row per ordered item, used mainly for product-category and revenue analysis
- `orders_level`: one row per order, used for monthly trends, customer distribution, delivery performance, and review analysis

The final cleaned data used by the dashboard is saved to:

```text
dashboard/main_data.csv
```

## Exploratory Data Analysis (EDA)

The EDA uses the cleaned tables at the appropriate grain so that orders containing multiple items are not accidentally counted multiple times.

### Explore: Question 1 (Product categories)

Revenue is calculated as the sum of item `price` values for delivered orders. Freight is excluded from the project revenue definition.

The category analysis includes:

- total revenue by category
- top and bottom categories
- revenue share
- cumulative revenue
- Pareto concentration

The analysis shows that revenue is concentrated in a relatively small group of categories, with the top five contributing about 40% of total revenue.

### Explore: Question 2 (Sales trend)

Monthly order and revenue trends are calculated from the order-level table.

The analysis compares:

- monthly order volume
- monthly revenue
- calendar-year patterns
- January–August year-over-year growth

The main trend is rapid expansion through 2017 followed by a higher but more stable monthly level during 2018.

### Explore: Question 3 (Customer location and delivery)

The state-level analysis includes:

- customer share by state
- average delivery time
- average review score
- late-order percentage
- review score by delivery-time group

State-level comparisons exclude very small samples when ranking delivery performance so that a tiny number of orders does not dominate the comparison.

## Visualization & Explanatory Analysis

The project follows a consistent visualization approach:

- Titles communicate the main message of each chart.
- Subtitles explain the metric, period, and data scope.
- Units are written directly on axes or labels.
- Data labels are added where they improve readability.
- Reference lines are used when a benchmark helps interpretation.
- State-level rankings avoid very small samples where necessary.
- Colors are used to distinguish context, focus values, and comparison groups rather than as decoration.

The dashboard and notebook use the same analytical definitions so that the interactive results remain consistent with the documented EDA.

## Analisis Lanjutan: RFM, Geospatial Analysis, dan Manual Grouping

Three additional techniques are applied without machine learning:

1. **RFM Analysis**  
   Customers are segmented using:
   - **Recency:** number of days since the customer's latest purchase, using **1 September 2018** as the reference date.
   - **Frequency:** number of orders placed by the customer.
   - **Monetary:** total amount spent.

   Recency and Monetary are scored from 1–5 using quantiles. Because most customers purchase only once, Frequency is handled with a business rule: customers with at least two orders are treated as repeat/loyal customers.

2. **Geospatial analysis**  
   Customer and seller locations are represented using ZIP-code coordinates from the geolocation table. The project visualizes customer density, summarizes performance by state and region, and measures approximate seller–customer distance using the haversine formula.

3. **Manual grouping/binning**  
   The analysis groups:
   - 27 Brazilian states into five regions: Norte, Nordeste, Centro-Oeste, Sudeste, and Sul.
   - Seller–customer distance into five manually defined ranges.

> **Note:** Coordinates are derived from the median point of each ZIP-code prefix in the `geolocation` table. Distance therefore represents an approximate straight-line distance, not an actual road route.

### RFM Analysis

The RFM analysis finds that repeat purchasing is limited:

- Only **3.0% of customers** (2,789 people) made more than one purchase.
- **97.0%** of customers purchased only once.
- Because Frequency is nearly constant for most customers, Recency and Monetary provide more separation between customer groups.
- The **high-value at-risk** segment represents **22.2% of customers** and contributes **41.9% of revenue**.
- The **high-value new** segment represents **15.5% of customers** and contributes **29.3% of revenue**.
- The **loyal** segment is only **3.0% of customers** but contributes **5.5% of revenue**.
- Customers with the highest Recency score have made a purchase within roughly the last **94 days** of the reference date.
- The highest Monetary score corresponds to customers spending at least approximately **R$ 180**.

The main pattern is that most customers are one-time buyers, while a relatively small group contributes a substantial share of revenue.

### Geospatial Analysis and Manual Grouping

The geographic analysis shows:

- Customer orders are concentrated across the **Southeast and South** of Brazil, especially around SP, RJ, and MG.
- The **Sudeste** region contains about **69% of customers**.
- Average delivery time is approximately **10.7 days** in Sudeste and **22.6 days** in Norte.
- Delivery time rises as seller–customer distance increases.
- Average delivery time grows from about **6.5 days** for distances under 100 km to approximately **21.2 days** for distances above 2,000 km.
- Freight represents about **12% of item price** at the shortest distances and about **23%** at the longest distances.
- About **16% of orders** with usable coordinates travel more than 1,000 km.
- The correlation between seller–customer distance and delivery time is approximately **r = +0.39**.
- Review scores also tend to be lower for the farthest distance group, averaging about **3.97** compared with **4.28** for the nearest group.

## Dashboard

The project includes an interactive Streamlit dashboard located at:

```text
dashboard/dashboard.py
```

The dashboard reads the cleaned dataset from:

```text
dashboard/main_data.csv
```

### Filters

The sidebar provides:

- Purchase date range
- Customer state
- Product category

Revenue is defined consistently with the notebook as the sum of item prices from delivered orders, excluding freight.

### Dashboard sections

#### 1 · Kategori

Shows:

- Revenue by product category
- Top and bottom categories
- Cumulative revenue / Pareto curve
- Revenue contribution of leading categories

#### 2 · Tren

Shows:

- Monthly order volume
- Monthly revenue
- Peak month
- Comparison between available calendar months in 2017 and 2018
- Year-over-year changes

#### 3 · Wilayah & Pengiriman

Shows:

- Customer share by state
- Average delivery time by state
- Review score by delivery-time group
- Late-order percentage
- Comparison of delivery performance and customer satisfaction

#### 4 · RFM

Shows:

- Customer segment distribution
- Customer share vs. revenue share
- Average spend by segment
- Repeat-customer rate

#### 5 · Geospatial

Shows:

- Customer density visualization
- Regional delivery and review summaries
- Seller–customer distance groups
- Interactive state-level Folium map
- Location-based delivery and customer-satisfaction metrics

## Project Structure

```text
submission_ecommerce/
├── dashboard/
│   ├── dashboard.py          # Streamlit dashboard
│   └── main_data.csv         # Cleaned analysis-ready dataset
├── data/
│   ├── customers_dataset.csv
│   ├── geolocation_dataset.csv.gz
│   ├── order_items_dataset.csv
│   ├── order_payments_dataset.csv
│   ├── order_reviews_dataset.csv
│   ├── orders_dataset.csv
│   ├── product_category_name_translation.csv
│   ├── products_dataset.csv
│   └── sellers_dataset.csv
├── .streamlit/
│   └── config.toml           # Streamlit configuration/theme
├── .gitignore
├── notebook.ipynb             # Full data cleaning, EDA, and analysis
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── url.txt                   # Deployed dashboard URL
```

## Requirements

- Python 3.11
- pandas >= 2.0
- NumPy >= 1.24
- Matplotlib >= 3.7
- Folium >= 0.15
- Streamlit >= 1.30

The exact dependencies are listed in `requirements.txt`.

## Local Setup

### Option 1 — Anaconda

Create and activate the environment:

```bash
conda create --name main-ds python=3.11
conda activate main-ds
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Option 2 — Python virtual environment

Open a terminal in the project root:

```bash
cd submission_ecommerce
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Notebook

Open `notebook.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.

For local Jupyter:

```bash
pip install jupyter
jupyter notebook
```

Run the notebook from the project root so the relative paths to `data/` work correctly.

The notebook will:

1. Load the nine source tables.
2. Assess data quality.
3. Clean and validate the tables.
4. Build the `sales` and `orders_level` analytical tables.
5. Calculate delivery and customer metrics.
6. Perform exploratory analysis.
7. Generate explanatory visualizations.
8. Run RFM, geospatial, and manual grouping analyses.
9. Save the cleaned data to `dashboard/main_data.csv`.

## Run the Streamlit Dashboard

From the project root, run:

```bash
streamlit run dashboard/dashboard.py
```

Streamlit will normally open the dashboard at:

```text
http://localhost:8501
```

Running the command from the project root is recommended so that the relative data path remains consistent.

## Deploy to Streamlit Community Cloud

The project structure is prepared for deployment.

Open [Streamlit Community Cloud](https://share.streamlit.io/) and create a new app using the repository containing this project.

Set:

```text
Repository : <your repository>
Branch     : main
Main file  : dashboard/dashboard.py
```

Then deploy the application.

After deployment, place the public application URL in:

```text
url.txt
```

## Repository Notes

The notebook is the source of the cleaned data used by the dashboard. If the raw data or cleaning logic changes, rerun the notebook and update:

```text
dashboard/main_data.csv
```

before deploying the dashboard again.

The dashboard operates from the cleaned item-level dataset while converting the current filtered data into an order-level table where order-based metrics are required. This prevents orders containing multiple items from being counted multiple times in delivery, customer, and review analysis.

## Limitations

- The dataset represents historical marketplace activity from 2016–2018 and does not describe current Olist operations.
- The main business analysis uses January 2017–August 2018 because this is the period with complete monthly coverage.
- Correlation describes statistical association and does not establish causation.
- The dataset does not contain a complete operational route history, so seller–customer distance is an approximate straight-line measure.
- Geolocation coordinates are represented by ZIP-code-level points rather than exact customer or seller addresses.
- Small state samples can be unstable, so some rankings use a minimum-order threshold.
- Review text is not analyzed in the main project.
- Some missing and inconsistent records are excluded when the required analytical fields are not valid.
- Outliers are generally retained when they appear plausible as real transactions.
- Manual groups are analytical categories created for interpretation and are not official business standards.
- The RFM reference date is fixed at **1 September 2018**.
- The analysis does not use predictive machine learning.

## Conclusion

- **Question 1 (categories):** Health Beauty, Watches Gifts, Bed Bath Table, Sports Leisure, and Computers Accessories are the five largest revenue categories and together contribute **39.9%** of total revenue. Only **18 of 73 categories** account for the first 80% of cumulative revenue.
- **Question 2 (trend):** monthly orders peak at **7,288 in November 2017**. Comparing January–August 2018 with January–August 2017, orders rise by **139.9%** and revenue by **141.1%**, while the magnitude of monthly YoY growth becomes smaller later in the period.
- **Question 3 (location and delivery):** SP contains **41.9% of customers**. Among states with sufficient order volume, average delivery time ranges from **8.7 days in SP** to **26.4 days in AM**. Late orders receive an average score of **2.57**, compared with **4.29** for on-time orders.
- **Additional analysis:** repeat customers represent only **3.0%** of unique customers. Geographically, customers and orders are concentrated in Southeast and South Brazil, while longer seller–customer distances are associated with longer delivery times and a higher freight share.

## Action Items

1. **Monitor high-revenue categories.** The top categories contribute a large share of total revenue, so category-level sales and inventory performance should be monitored separately rather than treating the entire catalog as a single segment.
2. **Prepare for high-volume months.** November 2017 is the peak month in the analyzed period. Monthly order history can be used to plan logistics capacity and campaign periods.
3. **Investigate delivery performance by state.** States with long delivery times can be examined further for seller distribution, carrier coverage, and estimated-delivery accuracy.
4. **Use delivery performance as a customer-experience metric.** The difference in review scores between late and on-time orders suggests that delivery timeliness and customer satisfaction should be monitored together.
5. **Develop retention analysis.** Because only a small share of customers place repeat orders, RFM segments can be used to study repeat-purchase behavior and customer value.
6. **Review geographic logistics patterns.** Long-distance orders have higher delivery times and freight shares, making seller location and regional fulfillment important variables for further analysis.

## Data Attribution

The dataset is the **Brazilian E-Commerce Public Dataset by Olist**, distributed through Kaggle:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Dataset license: **CC BY-NC-SA 4.0**.

Please retain the original dataset attribution and refer to the source page for the complete license terms when reusing or redistributing the data.

## Author

**Raihan Muzhaffar Athallah**

Final Project — Fundamental Data Analysis
#   F i n a l - P r o j e c t _ F u n d a m e n t a l - D a t a - A n a l y s i s _ E - C o m m e r c e - P u b l i c - D a t a s e t  
 