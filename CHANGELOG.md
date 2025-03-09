# Changelog

## [Unreleased]
### Added
- Implemented a federated learning system using Kafka.
- Added `FederatedServer` class to manage the global model and client updates.
- Added `FederatedClient` class to train a local model on the CIFAR-10 dataset.
- Implemented model initialization using a pre-trained CIFAR-10 architecture.
- Added logging for better debugging and tracking of model updates.

### Changed
- Modified the server to dynamically assign client IDs upon receiving updates.
- Updated the client to send a registration request to the server upon startup.
- Changed the model serialization and deserialization methods to ensure compatibility with Kafka.

### Fixed
- Corrected the model weight handling to ensure proper communication between the server and clients.
- Ensured that the client can handle receiving the model from the server correctly.

## [0.0.1] - 2025-03-09
### Initial Release
- Basic implementation of federated learning with Kafka.
- Server and client communication established.
- CIFAR-10 dataset used for training.