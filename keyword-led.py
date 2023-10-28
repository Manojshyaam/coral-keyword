import argparse
import sys
import model
import numpy as np
import coral_voice_detection
from periphery import GPIO

# Initialize the LED and Button
led = GPIO("/dev/gpiochip0", 39, "out")  # GPIO pin 39
button = GPIO("/dev/gpiochip0", 13, "in")  # GPIO pin 13

# Flag to track keyphrase detection
keyphrase_detected = False

# Define a callback for result processing
def result_callback(result, commands, labels, top=3):
    nonlocal keyphrase_detected
    top_results = np.argsort(-result)[:top]
    for p in range(top):
        l = labels[top_results[p]]
        if l in commands.keys():
            threshold = commands[labels[top_results[p]]]["conf"]
        else:
            threshold = 0.5
        if top_results[p] and result[top_results[p]] > threshold:
            sys.stdout.write("\033[1m\033[93m*%15s*\033[0m (%.3f)" % (l, result[top_results[p]]))
            # Check if the keyphrase "switch on" is detected
            if l == "switch on" and result[top_results[p]] > threshold:
                keyphrase_detected = True
                led.write(True)  # Turn on the LED
        elif result[top_results[p]] > 0.005:
            sys.stdout.write(" %15s (%.3f)" % (l, result[top_results[p]))
    sys.stdout.write("\n")

def main():
    parser = argparse.ArgumentParser()
    model.add_model_flags(parser)
    args = parser.parse_args()
    interpreter = model.make_interpreter(args.model_file)
    interpreter.allocate_tensors()
    mic = args.mic if args.mic is None else int(args.mic)

    # Initialize the Coral Voice Detection model (please set up the model separately)
    coral_voice_detection.setup_voice_detection()

    try:
        while True:
            # Capture audio and process it
            audio_data = model.capture_audio(mic, sample_rate_hz=int(args.sample_rate_hz))
            model.process_audio(interpreter, audio_data)
            model.classify_audio(mic, interpreter,
                                labels_file="config/labels_gc2.raw.txt",
                                result_callback=result_callback,
                                sample_rate_hz=int(args.sample_rate_hz),
                                num_frames_hop=int(args.num_frames_hop))

            # Check if the keyphrase is detected and turn off the LED if necessary
            if keyphrase_detected:
                led.write(False)  # Turn off the LED
                keyphrase_detected = False

    except KeyboardInterrupt:
        pass
    finally:
        led.write(False)  # Turn off the LED
        led.close()

if __name__ == "__main__":
    main()
