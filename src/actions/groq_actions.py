import json
import logging
import os
from dotenv import load_dotenv
from src.action_handler import register_action

logger = logging.getLogger("actions.groq_actions")

systemPrompt = "You are an AI financial expert specializing in real-world assets (RWA), cryptocurrencies, stocks, forex, asset diversification, liquidation, and decentralized finance (DeFi). Your goal is to provide accurate, insightful, and actionable information to users seeking guidance on trading, investment strategies, risk management, and financial decision-making.",
"Your knowledge includes but is not limited to:",
"Real-World Assets (RWA): Tokenization of physical assets, RWA-backed cryptocurrencies, and their role in DeFi.",
"Cryptocurrencies: Blockchain technology, major cryptocurrencies (Bitcoin, Ethereum, etc.), altcoins, stablecoins, and crypto trading strategies.",
"Stocks: Equity markets, stock analysis (fundamental and technical), portfolio management, and trading strategies.",
"Forex: Currency trading, exchange rates, forex market analysis, and risk management.",
"Asset Diversification: Strategies for spreading risk across asset classes, sectors, and geographies.",
"Liquidation: Understanding liquidation processes in trading, margin trading, and DeFi protocols.",
"DeFi: Decentralized exchanges (DEXs), yield farming, staking, liquidity pools, and smart contracts.",
"When responding to users:",
"Provide clear, concise, and well-structured explanations. Highlight risks and potential rewards associated with any strategy or decision. Stay updated on the latest trends, regulations, and market developments in the financial world.",
"Offer actionable advice tailored to the user's level of expertise (beginner, intermediate, or advanced).",
"Your tone should be professional, approachable, and educational, ensuring users feel confident in their understanding of complex financial concepts."

@register_action("generate-text")
def generate_text(agent, **kwargs):
    """Generate text using Groq models.
    This is a passthrough to groq_connection.generate_text().
    Add your custom logic here if needed!
    """
    try:
        prompt = kwargs.get("prompt")
        # system_prompt = kwargs.get("system_prompt")
        system_prompt = kwargs.get(systemPrompt)
        model = kwargs.get("model")
        temperature = kwargs.get("temperature")

        if not prompt or not system_prompt:
            logger.error("Missing required parameters: prompt and system_prompt")
            return None

        # Direct passthrough to connection method - add your logic before/after this call!
        return agent.connection_manager.connections["groq"].generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature
        )

    except Exception as e:
        logger.error(f"Failed to generate text: {str(e)}")
        return None

@register_action("check-model")
def check_model(agent, **kwargs):
    """Check if a specific Groq model is available.
    This is a passthrough to groq_connection.check_model().
    Add your custom logic here if needed!
    """
    try:
        model = kwargs.get("model")
        
        if not model:
            logger.error("No model name provided")
            return None

        # Direct passthrough to connection method - add your logic before/after this call!
        return agent.connection_manager.connections["groq"].check_model(
            model=model
        )

    except Exception as e:
        logger.error(f"Failed to check model availability: {str(e)}")
        return None

@register_action("list-models")
def list_models(agent, **kwargs):
    """List all available Groq models.
    This is a passthrough to groq_connection.list_models().
    Add your custom logic here if needed!
    """
    try:
        # Direct passthrough to connection method - add your logic before/after this call!
        return agent.connection_manager.connections["groq"].list_models()

    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        return None

@register_action("token-query")
def token_query(agent, **kwargs):
    """Process natural language token queries and return token information"""
    try:
        prompt = kwargs.get("prompt")
        if not prompt:
            logger.error("No prompt provided")
            return None

        # First, use Groq to extract token information from the prompt
        system_prompt = """
        Extract token information from the user query. Look for:
        1. Token symbol or name
        2. Chain ID (if mentioned)
        3. DEX ID (if mentioned)
        Format the response as JSON with fields: token_symbol, chain_id, dex_id
        If chain_id or dex_id are not mentioned, set them to null.
        """
        
        query_info = agent.connection_manager.connections["groq"].generate_text(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        try:
            # Parse the JSON response
            query_params = json.loads(query_info)
            
            # Get token information using the extracted parameters
            return agent.perform_action("get-token-info", {
                "token_symbol": query_params.get("token_symbol"),
                "chain_id": query_params.get("chain_id"),
                "dex_id": query_params.get("dex_id")
            })
            
        except json.JSONDecodeError:
            logger.error("Failed to parse Groq response as JSON")
            return None

    except Exception as e:
        logger.error(f"Token query failed: {str(e)}")
        return None