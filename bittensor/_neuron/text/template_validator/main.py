import bittensor
if __name__ == "__main__":
    
    neuron = bittensor.neurons.template_validator.neuron()
    with neuron:
        neuron.run()
