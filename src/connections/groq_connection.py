import logging
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv, set_key
from openai import OpenAI
from src.connections.base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.groq_connection")

class GroqConnectionError(Exception):
    """Base exception for Groq connection errors"""
    pass

class GroqConfigurationError(GroqConnectionError):
    """Raised when there are configuration/credential issues"""
    pass

class GroqAPIError(GroqConnectionError):
    """Raised when Groq API requests fail"""
    pass

class GroqConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None

    @property
    def is_llm_provider(self) -> bool:
        return True

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Groq configuration from JSON"""
        required_fields = ["model"]
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")
            
        if not isinstance(config["model"], str):
            raise ValueError("model must be a string")
            
        return config

    def register_actions(self) -> None:
        """Register available Groq actions"""
        self.actions = {
            "generate-text": Action(
                name="generate-text",
                parameters=[
                    ActionParameter("prompt", True, str, "The input prompt for text generation"),
                    ActionParameter("system_prompt", True, str, "System prompt to guide the model"),
                    ActionParameter("model", False, str, "Model to use for generation"),
                    ActionParameter("temperature", False, float, "A decimal number that determines the degree of randomness in the response.")
                ],
                description="Generate text using Groq models"
            ),
            "check-model": Action(
                name="check-model",
                parameters=[
                    ActionParameter("model", True, str, "Model name to check availability")
                ],
                description="Check if a specific model is available"
            ),
            "list-models": Action(
                name="list-models",
                parameters=[],
                description="List all available Groq models"
            ),
            "get-token-info": Action(
                name="get-token-info",
                parameters=[
                    ActionParameter("token_symbol", True, str, "Token symbol to search for"),
                    ActionParameter("system_prompt", True, str, "System prompt to guide the model"),
                    ActionParameter("chain_id", False, str, "Optional chain ID filter"),
                    ActionParameter("dex_id", False, str, "Optional DEX ID filter")
                ],
                description="Get detailed token information from Sonic"
            ),
            "token-query": Action(
                name="token-query",
                parameters=[
                    ActionParameter("prompt", True, str, "Natural language token query")
                ],
                description="Process natural language queries about tokens and return detailed information"
            ),
        }

    def _get_client(self) -> OpenAI:
        """Get or create Groq client"""
        if not self._client:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise GroqConfigurationError("Groq API key not found in environment")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        return self._client

    def configure(self) -> bool:
        """Sets up Groq API authentication"""
        logger.info("\n🤖 GROQ API SETUP")

        if self.is_configured():
            logger.info("\nGroq API is already configured.")
            response = input("Do you want to reconfigure? (y/n): ")
            if response.lower() != 'y':
                return True

        logger.info("\n📝 To get your Groq API credentials:")
        logger.info("Go to https://console.groq.com")
        
        api_key = input("\nEnter your Groq API key: ")

        try:
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            set_key('.env', 'GROQ_API_KEY', api_key)
            
            # Validate the API key by trying to list models
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            client.models.list()

            logger.info("\n✅ Groq API configuration successfully saved!")
            logger.info("Your API key has been stored in the .env file.")
            return True

        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def is_configured(self, verbose = False) -> bool:
        """Check if Groq API key is configured and valid"""
        try:
            load_dotenv()
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                return False

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            client.models.list()
            return True
            
        except Exception as e:
            if verbose:
                logger.debug(f"Configuration check failed: {e}")
            return False

    def generate_text(self, prompt: str, system_prompt: str, model: str = None, **kwargs) -> str:
        """Generate text using Groq models"""
        try:
            client = self._get_client()
            csystem_prompt="You are an AI financial expert specializing in real-world assets (RWA), cryptocurrencies, stocks, forex, asset diversification, liquidation, and decentralized finance (DeFi). Your goal is to provide accurate, insightful, and actionable information to users seeking guidance on trading, investment strategies, risk management, and financial decision-making.", "Your knowledge includes but is not limited to:", "Real-World Assets (RWA): Tokenization of physical assets, RWA-backed cryptocurrencies, and their role in DeFi.", "Cryptocurrencies: Blockchain technology, major cryptocurrencies (Bitcoin, Ethereum, etc.), altcoins, stablecoins, and crypto trading strategies.", "Stocks: Equity markets, stock analysis (fundamental and technical), portfolio management, and trading strategies.", "Forex: Currency trading, exchange rates, forex market analysis, and risk management.", "Asset Diversification: Strategies for spreading risk across asset classes, sectors, and geographies.", "Liquidation: Understanding liquidation processes in trading, margin trading, and DeFi protocols.", "DeFi: Decentralized exchanges (DEXs), yield farming, staking, liquidity pools, and smart contracts.", "When responding to users:", "Provide clear, concise, and well-structured explanations. Highlight risks and potential rewards associated with any strategy or decision. Stay updated on the latest trends, regulations, and market developments in the financial world.", "Offer actionable advice tailored to the user's level of expertise (beginner, intermediate, or advanced).", "Your tone should be professional, approachable, and educational, ensuring users feel confident in their understanding of complex financial concepts."
            
            system_prompt = csystem_prompt
            
            # Use configured model if none provided
            if not model:
                model = self.config["model"]

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                
            )

            return completion.choices[0].message.content
            
        except Exception as e:
            raise GroqAPIError(f"Text generation failed: {e}")

    def check_model(self, model: str, **kwargs) -> bool:
        """Check if a specific model is available"""
        try:
            client = self._get_client()
            try:
                models = client.models.list()
                for groq_model in models.data:
                    if groq_model.id == model:
                        return True
                return False
            except Exception as e:
                raise GroqAPIError(f"Model check failed: {e}")
                
        except Exception as e:
            raise GroqAPIError(f"Model check failed: {e}")
        
    def get_token_info(self, token_symbol: str, chain_id: Optional[str] = None, dex_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get token information from Sonic connection.
        
        Args:
            token_symbol (str): Token symbol to search for
            chain_id (str, optional): Chain ID to filter results
            dex_id (str, optional): DEX ID to filter results
            
        Returns:
            List[Dict[str, Any]]: List of token information dictionaries
        """
        try:
            # Get token information from Sonic connection
            sonic_connection = self.connection_manager.connections.get("sonic")
            if not sonic_connection:
                raise ValueError("Sonic connection not available")

            tokens = sonic_connection.get_token_info(
                token_symbol=token_symbol,
                chain_id=chain_id,
                dex_id=dex_id
            )

            if not tokens:
                return []

            # Process and enhance token information with AI insights
            enhanced_tokens = []
            for token in tokens:
                # Generate AI insights about the token
                insights = self.generate_text(
                    prompt=f"Analyze this token pair:\nChain: {token['chain']}\nDEX: {token['dex']}\nName: {token['name']}\nSymbol: {token['symbol']}\nProvide a brief analysis of liquidity, trading volume, and potential risks.",
                    system_prompt="You are a DeFi expert. Analyze the token pair data and provide concise insights."
                )
                
                # Enhance token information with AI insights
                enhanced_token = {
                    **token,
                    "ai_analysis": insights
                }
                enhanced_tokens.append(enhanced_token)

            return enhanced_tokens

        except Exception as e:
            logger.error(f"Failed to get token information: {e}")
            return []

    def list_models(self, **kwargs) -> None:
        """List all available Groq models"""
        try:
            client = self._get_client()
            response = client.models.list().data
        
            model_ids= [model.id for model in response]

            logger.info("\nAVAILABLE MODELS:")
            for i, model_id in enumerate(model_ids, start=1):
                logger.info(f"{i}. {model_id}")
                    
        except Exception as e:
            raise GroqAPIError(f"Listing models failed: {e}")
    
    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Groq action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        # Explicitly reload environment variables
        load_dotenv()
        
        if not self.is_configured(verbose=True):
            raise GroqConfigurationError("Groq is not properly configured")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        # Call the appropriate method based on action name
        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)