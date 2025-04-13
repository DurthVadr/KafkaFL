import os
import argparse
import flwr as fl
from logging import INFO, basicConfig

# Configure logging
basicConfig(level=INFO)

def main():
    """Parse arguments and start server."""
    parser = argparse.ArgumentParser(description="Flower server using Kafka")
    parser.add_argument(
        "--broker", 
        type=str, 
        default="localhost:9094",
        help="Kafka broker address in the format host:port"
    )
    parser.add_argument(
        "--grpc", 
        action="store_true",
        help="Use gRPC instead of Kafka"
    )
    parser.add_argument(
        "--num-rounds", 
        type=int, 
        default=3,
        help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--min-clients", 
        type=int, 
        default=2,
        help="Minimum number of clients for training"
    )
    parser.add_argument(
        "--min-eval-clients", 
        type=int, 
        default=2,
        help="Minimum number of clients for evaluation"
    )
    parser.add_argument(
        "--min-available-clients", 
        type=int, 
        default=2,
        help="Minimum number of available clients"
    )
    parser.add_argument(
        "--fraction-fit", 
        type=float, 
        default=0.3,
        help="Fraction of clients to sample for training"
    )
    parser.add_argument(
        "--fraction-eval", 
        type=float, 
        default=0.2,
        help="Fraction of clients to sample for evaluation"
    )
    
    args = parser.parse_args()
    
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_eval=args.fraction_eval,
        min_fit_clients=args.min_clients,
        min_eval_clients=args.min_eval_clients,
        min_available_clients=args.min_available_clients,
    )
    
    # Start Flower server
    fl.server.start_server(
        server_address=args.broker,
        use_kafka=not args.grpc,
        strategy=strategy,
        config={"num_rounds": args.num_rounds}
    )


if __name__ == "__main__":
    main()
