# Climate Profiler App

Climate profiling application for analyzing temperature, precipitation, and extreme weather trends at country and subnational levels.

## Architecture

```
climate-profiler-app/
  src/           # Core modules
  tests/         # Unit and integration tests
  config/        # Configuration files
  docs/          # Documentation
```

## Modules

- **climate_fetcher**: Core climate fetcher functionality
- **trend_analyzer**: Core trend analyzer functionality
- **anomaly_detector**: Core anomaly detector functionality
- **profile_builder**: Core profile builder functionality
- **report_writer**: Core report writer functionality

## Quick Start

```bash
pip install -r requirements.txt
python -m climate_profiler_app.main
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT License - see LICENSE for details.
