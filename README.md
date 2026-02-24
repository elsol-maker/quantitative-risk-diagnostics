Quantitative Risk Diagnostics
This tool is designed to look past simple price changes to reveal the actual risk underlying an investment. It transforms raw market data into high-fidelity diagnostics, helping users understand whether an asset's performance is driven by quality or just market noise.

Key Capabilities
Risk vs. Noise: Instead of jagged price charts, the engine calculates a 21-day rolling volatility curve to show exactly when risk is expanding or contracting.

Smart Benchmarking: The system automatically identifies an asset's home market—such as mapping Greek stocks to the Athex Composite—ensuring that Systemic Beta calculations are mathematically accurate.

Performance Attribution: It uses the 10-Year Treasury Note as a dynamic risk-free rate to calculate the Annualized Sharpe Ratio, measuring how much return you get for every unit of risk taken.

AI Interpretation: A built-in integration with Gemini 2.0 Flash reads the quantitative data and writes a formal summary of the asset's risk profile.

Technical Foundation
The engine is built for accuracy and speed, utilizing vectorized operations to handle complex financial math across different timezones and asset classes.

Market Heat: A specialized 7-day tracker monitors the most and least volatile assets in Equities, Crypto, and Macro categories, with annualization factors adjusted for specific trading hours.

Data Integrity: Every query is sanitized and cached for one hour to maintain performance and stay within API limits.