# Contributing to MaskRefiner

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/neverbiss/MaskRefiner.git
cd MaskRefiner
pip install -e ".[dev]"
```

## Code Style

We use `black` for formatting and `isort` for import sorting:

```bash
black maskforge/ examples/ scripts/
isort maskforge/ examples/ scripts/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Reporting Issues

Please use GitHub Issues with a clear description and minimal reproduction steps.
