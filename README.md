# mmWave Breathing Monitor for Overdose Detection 

Setup Guide and System Documentation 

## July 15, 2026 

Project overview. This is a contactless breathing-rate monitoring system built around the TI XWR1843BOOST 77 GHz FMCW radar and the DCA1000EVM data-capture card. It performs multi-target tracking, per-target breathing frequency estimation, and validation of sinusoidal breathing patterns. The target application is detecting abnormal respiratory patterns (e.g., opioid overdose) in enclosed spaces such as public restrooms and bus shelters. 

# Contents 

|1<br>Har|dware setup|1|
|---|---|---|
|1.1|Required components. . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>1|
|1.2|Physical connections . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>1|
|1.3|XWR1843 antenna geometry<br>. . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>1|
|2<br>Soft|ware setup|2|
|2.1|Python environment . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>2|
|2.2|mmWave Studio session<br>. . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>2|
|2.3|Radar conguration parameters . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>2|
|3<br>Sign|al-processing pipeline|3|
|3.1|Stage 1  Raw frame parsing . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>3|
|3.2|Stage 2  TX separation<br>. . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>4|
|3.3|Stage 3  Range FFT with coherent averaging . . . . . . . . . . . .|. . . . . . . .<br>4|
|3.4|Stage 4  Inter-TX phase calibration<br>. . . . . . . . . . . . . . . . .|. . . . . . . .<br>4|
|3.5|Stage 5  Two-pipeline processing . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>4|
|3.6|Stage 6  Multi-target tracking. . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>4|
|3.7|Stage 7  Breathing rate estimation and validation. . . . . . . . . .|. . . . . . . .<br>5|
|3.8|Stage 8  Angle of arrival . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>5|
|4<br>Rep|ository structure and le roles|5|
|4.1|Core radar processing<br>. . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>5|
|4.2|Calibration<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>6|
|4.3|Recording and analysis . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>6|
|4.4|Lua conguration scripts . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>6|
|5<br>Typ|ical session workow|6|
|5.1|First-time setup (once per PC)<br>. . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>6|
|5.2|Session start (each testing session)<br>. . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>6|
|5.3|Recording a session . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>7|
|5.4|Post-session analysis . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>7|
|6<br>Rec|ording le format|7|
|6.1|<session>_tracks.csv<br>. . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>8|
|6.2|<session>_meta.json . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>8|
|6.3|<session>_signals.npz . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>8|



1 

mmWave Breathing Monitor 

2 

|7<br>Tro|ubleshooting|8|
|---|---|---|
|7.1|No data streaming . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>8|
|7.2|Fan display shows lateral streaks (wide angular spread). . . . . . . . . . . . .|. . .<br>8|
|7.3|RX0 and RX2 report identical values in diagnostic<br>. . . . . . . . . . . . . . .|. . .<br>9|
|7.4|Track disappears after a while even though the person is still there . . . . . .|. . .<br>9|
|7.5|Ghost reection at a range approximately twice a real target's range . . . . .|. . .<br>9|
|8<br>Kn|own limitations and future work|9|



# 1 Hardware setup 

## 1.1 Required components 

-  TI XWR1843BOOST evaluation board (77 GHz FMCW radar, 3 TX _×_ 4 RX) 

-  TI DCA1000EVM data-capture card 

-  USB cable to the XWR1843BOOST (for RS-232 control and rmware download) 

-  5 V barrel-jack power supply for the DCA1000 

-  Ethernet cable PC _↔_ DCA1000 

-  PC running mmWave Studio (version 02.01.01.00 tested) 

-  (Optional) Vernier Go Direct Respiration Belt for ground-truth data 

## 1.2 Physical connections 

1. Mount the XWR1843BOOST onto the DCA1000EVM via the 60-pin high-speed connector. 

2. Connect USB from PC to XWR1843BOOST (for control and rmware download). 

3. Connect Ethernet from PC to DCA1000EVM. 

4. Power the DCA1000EVM with the 5 V supply. The green LED should light steadily. 

5. Set the PC Ethernet adapter's IPv4 address to 192.168.33.30, subnet mask 255.255.255.0. 

## 1.3 XWR1843 antenna geometry 

The XWR1843BOOST antenna layout (from the TI datasheet) has one important subtlety: the board silkscreen labels dier from the names used in mmWave Studio. The correct mapping is: 

|Board label|Studio name|Role|
|---|---|---|
|TX1|Studio TX0|Azimuth left (on horizontal line)|
|TX2|Studio TX1|Azimuth right (on horizontal line, _λ_ from TX0)|
|TX3|Studio TX2|Elevation (oset up by _λ/_2)|



Table 1: XWR1843 TX antenna mapping. 

The azimuth virtual array is formed by combining Studio TX0 and TX2 with the four RX antennas (spaced _λ/_ 2 ), yielding a uniform 8-element ULA that supports meaningful angle-of-arrival estimation. Studio TX1 is the elevation antenna and is used only to boost SNR of the range prole. 

mmWave Breathing Monitor 

3 

# 2 Software setup 

## 2.1 Python environment 

Python 3.10+ is recommended. Required packages: 

<mark>pip install numpy scipy matplotlib pandas pip install openradar # DCA1000 parsing convention pip install godirect # optional , only if using the belt</mark> 

## 2.2 mmWave Studio session 

Every session begins with the following procedure in mmWave Studio: 

1. Open mmWave Studio. 

2. In the Connection tab, connect to the XWR1843BOOST board. Load the appropriate rmware images and conrm the device status reads XWR1843/ASIL-B/SOP:2/ES:2. 

3. In the same tab, connect to the DCA1000EVM. 

4. Open the Lua Shell tab. 

5. Load and run radar_config_3tx_4rx.lua via Browse. 

6. The script sets up the prole, chirps, and frame parameters, arms the DCA1000, and starts data streaming to UDP 192.168.33.30:4098. 

## 2.3 Radar conguration parameters 

The current radar_config_3tx_4rx.lua script uses: 

|Parameter|Description|Value|
|---|---|---|
|START_FREQ|Chirp start frequency (GHz)|77.0|
|IDLE_TIME|Time between chirps (s)|5|
|RAMP_END_TIME|Chirp ramp duration (s)|40|
|ADC_START_TIME|When ADC starts within chirp (s)|7.07|
|FREQ_SLOPE|Chirp slope (MHz/s)|78.020|
|ADC_SAMPLES|Samples per chirp|256|
|SAMPLE_RATE|ADC sample rate (ksps)|10 000|
|RX_GAIN|Receiver gain (dB)|30|
|CHIRP_LOOPS|Chirps per TX per frame|64|
|PERIODICITY|Frame period (ms)|20|
|TX order|Chirps within a loop|TX1, TX0, TX2|



Table 2: Radar conguration parameters. 

Derived quantities: 

-  Frame rate: 1000 _/_ 20 = 50 Hz 

-  Range resolution: _c/_ (2 _· B_ e ) _≈_ 7 _._ 5 cm/bin 

-  Maximum unambiguous range: 9.5 m (128 range-FFT bins) 

-  Coherent integration gain per frame: 10 log10(64) _≈_ 18 dB 

-  Bytes per frame: 192 _×_ 4 _×_ 256 _×_ 4 = 786 432 B (0.75 MB) 

-  Data rate: _≈_ 37 _._ 5 MB/s over UDP 

mmWave Breathing Monitor 

4 

# 3 Signal-processing pipeline 



<!-- Start of picture text -->
UDP capture (background thread)<br>LVDS-interleaved parser<br>parse_one_frame()<br>Separate by TX<br>separate_tx()<br>Range FFT (Hann window)<br>coherent avg over 64 chirps<br>Apply TX2 phase calibration<br>Background subtract Keep raw complex<br>(detection path) (phase path)<br>CFAR peak detection Phase extraction at<br>tracked bin<br>Kalman multi-target tracker<br>Phase unwrap → mm displacement<br>BPM estimation & sinusoidal validation<br>UI: fan display + BPM sidebar<br><!-- End of picture text -->

Figure 1: End-to-end signal processing pipeline. The two-pipeline design uses backgroundsubtracted data for detection but keeps the raw complex range FFT for phase extraction to avoid corrupting the breathing signal. 

## 3.1 Stage 1 Raw frame parsing 

Each UDP frame is 786 432 bytes containing 192 chirps ( 3 TX _×_ 64 loops). The DCA1000 uses an LVDS-interleaved format where every four consecutive int16 values encode the real and imaginary parts of two complex samples: 

raw [0] _,_ raw [1] _→_ Re[ _s_ 0 _, s_ 1] raw [2] _,_ raw [3] _→_ Im[ _s_ 0 _, s_ 1] 

This matches the openradar DCA1000.organize() convention. Prior to adopting this parser, an incorrect naive I, Q, I, Q interpretation caused RX0 and RX2 to appear identical in the data a subtle bug that took considerable diagnostic work to identify. 

## 3.2 Stage 2 TX separation 

The chirps within each loop are interleaved by TX. From the Lua chirp order (TX1, TX0, TX2): 

<mark>tx1_elev = frame [0::3] # elevation antenna tx0_azL = frame [1::3] # azimuth left tx2_azR = frame [2::3] # azimuth right</mark> 

mmWave Breathing Monitor 

5 

## 3.3 Stage 3 Range FFT with coherent averaging 

For each TX group, the 64 chirps at 256 ADC samples undergo: 

1. DC removal (subtract per-chirp mean) 

2. Hann window 

3. 256-point complex FFT 

4. Keep the lower 128 bins (positive frequencies) 

5. Coherent averaging across the 64 chirps 

Coherent averaging assumes the target barely moves within a 1.28-second frame (which is true for breathing chest displacement of a few millimetres) and provides an 18 dB SNR gain. 

## 3.4 Stage 4 Inter-TX phase calibration 

The TX0 and TX2 signal paths have dierent PCB routing lengths and therefore dierent phase osets. Without correction, the concatenated 8-element virtual array has a phase discontinuity at the _v_ 3 _→ v_ 4 boundary that causes the angle FFT to place a spurious peak near _±_ 90<sup>_◦_</sup> . 

The calibration script calibrate_tx_phase.py measures four per-RX correction factors (one for each RX channel, capturing PCB routing dierences) using a stationary reference target at boresight. The correction is stored in radar_phase_calibration.npz and applied per frame: 

<mark>if tx2_phase_correction is not None:</mark> 

<mark>rfft_tx2 = rfft_tx2 * tx2_phase_correction [:, np.newaxis]</mark> 

## 3.5 Stage 5 Two-pipeline processing 

A single background subtraction of complex values would corrupt the phase we need for millimetrescale chest displacement measurement. To avoid this, the pipeline forks after the range FFT: 

-  Detection path. Complex range FFT _−_ calibrated background. Magnitude of the result is passed to CFAR peak detection and Kalman tracking. Static clutter is suppressed so weak human returns become visible. 

-  Phase-extraction path. The raw complex range FFT (never background-subtracted) is passed to the tracker as well. At each track's range bin, the tracker extracts the raw complex phase, unwraps it across frames, and converts to displacement in millimetres: ∆ _r_ = ∆ _ϕ·λ/_ (4 _π_ ) . 

## 3.6 Stage 6 Multi-target tracking 

Conrmed peaks from CFAR feed the MultiTrackManager (multi_track_manager.py), which maintains up to four simultaneous Kalman-ltered tracks with: 

-  Constant-velocity state [ _bin, bin_<sup>˙</sup> ] 

-  Position gate scaled by Kalman uncertainty; oor 5 bins ( _≈_ 40 cm) 

-  Conrm-3, delete-150 (i.e., 3-frame conrmation, 3-second coasting tolerance at 50 Hz) 

-  Per-track phase unwrap using np.unwrap across the frame history 

mmWave Breathing Monitor 

6 

## 3.7 Stage 7 Breathing rate estimation and validation 

A 40-second sliding window of unwrapped phase feeds an FFT-based BPM estimator that searches the 0.040.6 Hz band (2.436 breaths/min). Every 5 seconds, three sinusoidal-quality metrics are computed for each conrmed track: 

|Metric|Denition|Tier-1|Tier-2|
|---|---|---|---|
|Energy ratio|Fraction of AC power inside breathing band|_≥_0_._35|_≥_0_._25|
|Peak concentration|Fraction of band power in the dominant FFT bin|_≥_0_._40|_≥_0_._30|
|Spectral SNR|Peak power / median out-of-band power (dB)|_≥_12|_≥_8|



Table 3: Sinusoidal validation thresholds. All three metrics must exceed their tier thresholds for the track to be classed as breathing. 

Tier-1 requires 10 seconds of phase history and catches normal breathing (636 BPM). Tier-2 activates at 25 seconds of history with relaxed thresholds to catch slow breathing (36 BPM), which is the clinically important range for opioid-induced respiratory depression. 

## 3.8 Stage 8 Angle of arrival 

For each conrmed track the tracker also produces an azimuth angle estimate. The eight azimuth virtual elements _{v_ 0 _, ..., v_ 3 _}_ = TX0 _×_ RX 0 _.._ 3 and _{v_ 4 _, ..., v_ 7 _}_ = TX2 _×_ RX 0 _.._ 3 undergo a 64-point angle FFT. The peak of the resulting spectrum gives the target's azimuth angle. 

# 4 Repository structure and le roles 

## 4.1 Core radar processing 

|File|Purpose|
|---|---|
|multi_breathing_monitor.py|Live monitoring application. Opens UDP socket, runs the full pipeline<br>(parse_→_range FFT_→_tracking_→_BPM), and displays the fan-shaped<br>range-angle heatmap with per-track BPM readouts.|
|multi_track_manager.py|MultiTrackManager and Track classes. Implements CFAR peak de-<br>tection, nearest-neighbour association, Kalman-ltered range tracking,<br>phase extraction, BPM estimation, and sinusoidal validation.|
|kalman_tracker.py|Constant-velocity Kalman lter over range-bin position. Position un-<br>certainty is capped to prevent runaway growth during coasting.|
|mimo_diagnostic.py|Standalone diagnostic that captures 50 frames from the DCA1000 and<br>reports RX cross-correlations, TX0 vs TX2 correlations, per-RX phase<br>ramps, and a full angle FFT spectrum. Used to verify the MIMO array<br>is producing spatially diverse data.|



## 4.2 Calibration 

|File|Purpose|
|---|---|
|calibrate_tx_phase.py|Captures 50 frames with a stationary reference target at boresight<br>(_∼_1.5 m directly in front of the radar) and computes per-RX TX2<br>phase corrections. Saves radar_phase_calibration.npz. Should be<br>re-run at the start of each testing session or after any change to the<br>radar's power state.|
|radar_phase_calibration.npz|(Generated le.)<br>Contains the four per-RX phase corrections that<br>align TX2's phase with TX0's. Loaded automatically by the monitor<br>and oine replay tools.|



mmWave Breathing Monitor 

7 

## 4.3 Recording and analysis 

|File|Purpose|
|---|---|
|record_ui.py|Minimal recording UI: shows number of tracked people and per-<br>track BPM, with START/STOP buttons and a description text<br>box.<br>Saves each session to recordings/<session_name>/ as<br>three les: tracks CSV, metadata JSON, and signals NPZ (con-<br>taining complex range FFTs).|
|offline_vital_sign_detection.py|Replays a recorded session through the same pipeline as the live<br>monitor. Interactive UI includes play/pause/seek and speed con-<br>trol. Also supports batch CSV export for A/B comparison of<br>algorithm parameters against the same recording.|
|plot_recording.py|Post-hoc plotter for recorded sessions. Interactive check-box col-<br>umn lets you select any combination of track and eld to overlay<br>on a time-series plot. Also renders a range-time waterfall from<br>the signals NPZ, and (with raw) an ADC viewer for full-mode<br>recordings.|



## 4.4 Lua conguration scripts 

|File|Purpose|
|---|---|
|radar_config_3tx_4rx.lua|Current production cong. Enables all three TX (TX0+TX1+TX2), all<br>four RX, 64 chirp loops, 256 ADC samples at 10 MHz sample rate, 20 ms|
||frame period. Runs in mmWave Studio's Lua shell after connecting to<br>the board.|



# 5 Typical session workow 

## 5.1 First-time setup (once per PC) 

1. Install Python dependencies (see 2.1). 

2. Congure PC Ethernet: 192.168.33.30, mask 255.255.255.0. 

3. Verify mmWave Studio is installed and can connect to the board. 

## 5.2 Session start (each testing session) 

1. Power on the DCA1000 (green LED should light). 

2. Open mmWave Studio, connect to the board and DCA1000. 

3. In the Lua Shell, browse to and run radar_config_3tx_4rx.lua. Wait for the print output conrming the frame conguration. 

4. Run python calibrate_tx_phase.py with a stationary reference target (e.g., a closed laptop) at boresight, 1.5 m away. The script will report the measured phase corrections and save them. 

5. Conrm the calibration succeeded (std across RX _<_ 30<sup>_◦_</sup> ). If it failed, reposition the target closer to true boresight and rerun. 

6. Ensure the target area is clear of people, then launch python multi_breathing_monitor.py or python record_ui.py depending on whether you're testing or capturing data. 

7. The application waits for you to press Enter, then does a 10-second background calibration. Keep the area empty during this window. 

mmWave Breathing Monitor 

8 

8. After background calibration completes, you can walk into the scene. Conrmed breathing tracks appear as coloured wedges on the fan display with their BPM shown in the sidebar. 

## 5.3 Recording a session 

1. Launch python record_ui.py. 

2. Type a session description in the text box (e.g., subject_A_normal). Maximum 20 characters; special characters are sanitised. 

3. Select recording mode: 

   -  summary ( _∼_ 40 MB/min): tracks CSV + complex range FFTs  sucient for oine replay and algorithm experimentation 

   -  full ( _∼_ 2.3 GB/min): also stores raw ADC data  required if you want to change range-FFT parameters later 

4. Click START RECORDING. 

## 5.4 Post-session analysis 

<mark># View tracks and range -time waterfall for the most recent session: python plot_recording.py</mark> 

<mark># View a specific session: python plot_recording.py --session session_20260706_143022_subject_a # Replay a recording through the current pipeline (interactive): python offline_vital_sign_detection .py --session <name > # Batch A/B: dump replayed tracks after tweaking algorithm parameters: python offline_vital_sign_detection .py --dump -csv reprocessed.csv</mark> 

# 6 Recording le format 

Each recording session produces a folder containing three les: 

## 6.1 <session>_tracks.csv 

One row per track observation per frame. Columns: 

|Column|Description|
|---|---|
|frame_idx|0-based frame counter within the session|
|frame_time_s|Seconds since session start|
|track_id|Persistent track identier (assigned by the tracker)|
|range_m|Radial distance in metres|
|angle_deg|Azimuth angle in degrees (positive = right of boresight)|
|bpm|Estimated breaths per minute|
|validated|1 if sinusoidal breathing pattern conrmed, else 0|
|breathing_score|Energy ratio in breathing band (01)|
|snr_db|Spectral SNR of breathing peak in dB|



mmWave Breathing Monitor 

9 

## 6.2 <session>_meta.json 

Session metadata: description, start time, radar conguration parameters, whether calibration was loaded, session duration, list of associated les. 

## 6.3 <session>_signals.npz 

NumPy archive containing: 

-  frame_times: array of per-frame timestamps, shape (n_frames,) 

-  range_profiles: bg-subtracted 1D range prole magnitude, shape (n_frames, n_bins) 

-  range_complex: complex range FFTs per virtual element before background subtraction, shape (n_frames, 12, n_bins). This is the key data for oine replay: the entire pipeline downstream of the range FFT can be recomputed from it. 

-  raw_adc (full mode only): the raw int16 DCA1000 samples, which allow re-computing the range FFT with dierent windowing or chirp-level processing. 

# 7 Troubleshooting 

## 7.1 No data streaming 

-  Verify the DCA1000 green LED is solid. 

-  Conrm PC Ethernet IP is 192.168.33.30. 

-  In mmWave Studio, run the Lua script and watch the Output tab for any red error messages. RfInit in particular takes a few seconds and must succeed before frames start. 

## 7.2 Fan display shows lateral streaks (wide angular spread) 

Usually caused by inter-TX phase oset. Symptoms: 

-  Real targets appear at _±_ 90<sup>_◦_</sup> (edges of the fan) 

-  HPBW _>_ 25<sup>_◦_</sup> even for point targets 

Fix: run calibrate_tx_phase.py again with a good stationary target at boresight. 

## 7.3 RX0 and RX2 report identical values in diagnostic 

Indicates an LVDS-format mismatch between the Lua cong and the Python parser. The Lua script must set DataPathConfig(513, 1216644097, 0) and LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0) (2 LVDS lanes) for the openradar-style parser to work correctly. 

## 7.4 Track disappears after a while even though the person is still there 

Multiple possible causes: 

-  Adaptive background enabled and absorbing the target. Set ADAPTIVE_BG_ENABLED = False in multi_breathing_monitor.py for static environments. 

-  Phase history buer too short. Should be _≥_ 60 seconds for a 40-second BPM window at 50 Hz frame rate. 

-  Delete-frames threshold too aggressive. Default is 150 frames (3 seconds), which is generally reasonable. 

mmWave Breathing Monitor 

10 

## 7.5 Ghost reection at a range approximately twice a real target's range 

This is a multipath artifact: radar _→_ subject _→_ wall _→_ subject _→_ radar. Options in decreasing order of eectiveness: 

1. Tilt the radar down by 1520<sup>_◦_</sup> to reduce wall illumination. 

2. Place RF-absorbing material (heavy blanket, foam) on the wall behind the target. 

3. Point the radar toward open space rather than a wall. 

4. Implement ghost suppression in software (match tracks with the same BPM and angle at approximately double the range and reject the farther one). 

# 8 Known limitations and future work 

-  Multipath ghosts remain a challenge in conned spaces. The ghost-suppression algorithm described in 7.5 is not yet integrated. 

-  Sub-boresight resolution is limited by the 8-element virtual array to approximately 14<sup>_◦_</sup> halfpower beamwidth. A person's angular width is comparable to this, so two people less than _∼_ 30<sup>_◦_</sup> apart at the same range may merge into a single track. 

-  Elevation estimation is not currently used. The XWR1843's TX1 (elevation antenna) is captured for range-prole SNR but its phase data is discarded. 

-  No motion compensation: if the subject moves several centimetres during the 40-second BPM window, the phase unwrap can pick up the motion signal and corrupt the BPM estimate. A rst-order x would be to detrend the phase before the spectral analysis. 

-  Belt integration is optional and not required for radar-only breathing rate measurements; it provides ground truth during validation studies. 

