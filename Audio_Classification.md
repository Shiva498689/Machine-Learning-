<h4><br>
Aim:
The model classifies audio clips into multiple sound classes using a deep convolutional neural <br>network (ResNet152) — a powerful architecture originally designed for image classification, now <br>adapted for audio by converting waveforms into spectrogram images.

 Configuration:

<br>Sampling Rate (SR) = 22,050 Hz

<br>Clip Length = 5 seconds (audio is padded or trimmed to this)

<br>Mel Filterbanks (N_MELS) = 128

<br>Batch Size = 16 (ResNet152 is memory-heavy)

<br>Learning Rate = 1e-4

<br>Epochs = 40

<br>Optimizer = AdamW

<br>Loss = CrossEntropy with Label Smoothing (0.1)

<br>Early Stopping = after 6 epochs of no improvement

<br>A fixed random seed ensures reproducibility across runs.<br><br>

  Audio Preprocessing
<br>(a) load_and_fix_length()

<br>Loads an audio file with librosa.

<br>Trims silence from start and end.

<br>Pads or truncates it to a fixed 5-second length.

<br>Returns a normalized float32 waveform.

(b) extract_log_mel()

<p>Converts the waveform into a Mel Spectrogram (frequency vs time representation).

Applies power_to_db → converts power to decibel scale (logarithmic perception).

Normalizes spectrogram values to [0, 1] range.

Converts it to a 128×time matrix suitable for CNN input.</p>

<br> 3. Dataset & DataLoader<br>

<p>A custom AudioDataset:<br>

Loads each file → extracts Mel spectrogram → stacks it into 3 channels (R, G, B), mimicking image input (since ResNet expects RGB images).

Converts it into a torch tensor.

Returns (mel_tensor, label) for training, or (mel_tensor, filename) for testing.

Dataloaders are created for:

Train, Validation, and Test sets with appropriate batching and shuffling.</p><br>

 <br>Model Architecture: ResNet152<br>

<p>Pretrained ResNet152 from ImageNet is loaded (models.resnet152(weights='IMAGENET1K_V1')).

The final fully connected layer (model.fc) is replaced with a new Linear layer matching the number of audio classes.

This way, the model leverages transfer learning — reusing powerful visual features learned on ImageNet for sound spectrograms.</p><br>

 <br> Training Process<br>

<p>The model is trained using CrossEntropyLoss with label smoothing (helps generalization).

AdamW optimizer (better weight decay control).

Learning rate scheduler (ReduceLROnPlateau) lowers LR when validation accuracy plateaus.</p><br>

<br>Each epoch:<br>

<p>Train phase: forward + backward propagation, optimizer step.

Validation phase: compute accuracy & loss on unseen data.

Save model weights if validation accuracy improves.

Stop early if no improvement for 6 epochs.

After training, the best model weights are stored in "resnet_152_model.pth".</p>

<br> Testing & Predictions<br>

<p>Loads the best model checkpoint.

Predicts labels for all test files.

Uses the inverse LabelEncoder to convert numeric predictions → class names.

Exports results into a CSV:
test_predictions_resnet.csv</p>


 <br>Key Advantages<br>
<p>
*** Transfer Learning — ResNet152 brings strong visual pattern recognition, useful for spectrograms.
*** Label Smoothing — Reduces overfitting.
*** Early Stopping + Scheduler — Stabilizes training and prevents wasted computation.
*** Normalized, Log-scaled Features — Makes training more efficient.
</p>

<br>In Essence<br>
<p>
This pipeline:

Converts raw audio → spectrogram images.

Fine-tunes a pretrained ResNet152 image model.

Achieves robust multi-class audio classification performance.

Outputs predictions ready for submission or evaluation.</p>