# ========================================================
# Stage 1: Build Blazor App (.NET 10)
# ========================================================
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS blazor-build
WORKDIR /src

# Copy project file and restore dependencies
COPY frontend/BlazorApp/BlazorApp.csproj ./frontend/BlazorApp/
RUN dotnet restore frontend/BlazorApp/BlazorApp.csproj

# Copy all Blazor source files and publish release
COPY frontend/BlazorApp/ ./frontend/BlazorApp/
RUN dotnet publish frontend/BlazorApp/BlazorApp.csproj -c Release -o /app/blazor --no-restore

# ========================================================
# Stage 2: Final Production Runtime (ASP.NET 10 + Python 3)
# ========================================================
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
WORKDIR /app

# Install Python 3, pip, curl, ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy backend, demo data, and published Blazor app
COPY backend/ ./backend/
COPY demo_data/ ./demo_data/
COPY DEMO_PACK/ ./DEMO_PACK/
COPY --from=blazor-build /app/blazor ./blazor/

# Environment settings
ENV PORT=5000
ENV QC_PUBLIC_DEMO=false
ENV QcApi__BaseUrl=http://127.0.0.1:8000
ENV ASPNETCORE_ENVIRONMENT=Production
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Entrypoint script: starts Python backend internally on 8000, then starts Blazor UI on $PORT
RUN printf '#!/bin/sh\n\
PORT=8000 HOST=127.0.0.1 python3 /app/backend/app.py &\n\
sleep 2\n\
cd /app/blazor && exec dotnet BlazorApp.dll\n' > /app/start.sh && chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]
