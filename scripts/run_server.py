#!/usr/bin/env python
"""Run the Flask backend from package hidden_gems.backend.app"""
from hidden_gems.backend.app import app


def main(argv=None):
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
