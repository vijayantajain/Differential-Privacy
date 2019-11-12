# Differential-Privacy

A library that provides an interface to provide differentially private queries to a dataset/database

Currently, the library provides basic queries, such as count and average for a particular column. In  future the goal will be to be able to execute SQL queries

## Installation Instructions

Following are the instructions to install the package -

1. Clone the repository using `git clone https:\github.com\vijayantajain\Differential-Privacy`
2. Create a virtual environment (Optional, but highly recommended)
3. Install all the dependencies using the command `pip install -r requirements.txt`

## Running the Script

To test the functionalities, run the `foo.py` script as follows -  

```bash
python foo.py
```

The script executes differentially private queries on the [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/Adult)

The noise added to each query is using the Laplacian mechnanism which is centered at `0` and distribution parameter is `1\epsilon`. You can change the epsilon value in the `foo.py`.

You should run the script multiple times to see the difference in returned values.

## Modifying the Script

You can also test the functions on other categorical and continuous values by changing the `column_name` and providing the categories as list.
