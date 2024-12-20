from pyopenie import OpenIE5
extractor = OpenIE5('http://localhost:9000')

extractions = extractor.extract("The U.S. president Barack Obama gave his speech to thousands of people.")

print(extractions)