import uvicorn
import src

if __name__ == '__main__':
    #curl -X POST http://127.0.0.1:8000/api/v1/loan/predict -H "Content-Type: application/json" -d "{\"emp_length\":\"5 years\",\"int_rate\":14,\"annual_inc\":200000,\"fico_range_high\":700,\"loan_amnt\":9000}"
    uvicorn.run("src.app:app",host="0.0.0.0",port=8000)