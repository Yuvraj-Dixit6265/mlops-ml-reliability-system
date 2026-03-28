FROM python:3.11

WORKDIR /app

# Install system dependencies (for whylogs)
RUN apt-get update && apt-get install -y build-essential

# Copy only requirements first (for caching)
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Now copy rest of project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]