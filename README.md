# word2box-for-reccomendation

## Setup

### Requirements
- docker
- docker-compose

### Demo
![](https://private-user-images.githubusercontent.com/74698040/244950346-decbc303-66a7-4e0d-91cd-9ff58562be48.webm?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXkiOiJrZXkxIiwiZXhwIjoxNjg2NDk5NjM3LCJuYmYiOjE2ODY0OTkzMzcsInBhdGgiOiIvNzQ2OTgwNDAvMjQ0OTUwMzQ2LWRlY2JjMzAzLTY2YTctNGUwZC05MWNkLTlmZjU4NTYyYmU0OC53ZWJtP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQUlXTkpZQVg0Q1NWRUg1M0ElMkYyMDIzMDYxMSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyMzA2MTFUMTYwMjE3WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9ODk1ZTM2ODY1NWQxYWVhNGJlNGRkNmEwZDVmNTI1NzAwNDI3NDYwNGViNWRjZjMxMGU0MGI1YjViNzkzM2Q5YyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.PTA5-XDj-orox3VMXOhlttI-6D51gpj8ctTNcGRPgyw)

### Build
```
docker-compose up gpu
```

## Train Word2Box
```
cd word2box
sh run.sh
```

## Run Streamlit
```
streamlit run app.py
```


