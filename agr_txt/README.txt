agr_txt — inspect Ableton .agr groove files
===========================================

WHAT THIS IS
------------
An .agr file is an Ableton "groove" (or clip) saved as XML and then gzip-
compressed, so you can't read it directly. agr_to_txt.py decompresses it and
writes the XML out as a readable .txt, so you can see exactly what timing and
velocity data the groove contains (the same data the Björk Cube app folds onto
its 16-slot step grid).


HOW TO USE
----------
1. Copy one or more .agr files into this folder (agr_txt/).
2. Run the script from inside this folder:

       python3 agr_to_txt.py

   (Plain Python 3 — no extra packages needed.)
3. For every  <name>.agr  it writes  <name>.txt  next to it, containing the
   pretty-printed XML. Re-running overwrites the .txt files.

The script prints one line per file, e.g.:
       wrote AmenBreak.txt  (5059 chars, gzipped source)


WHAT THE FIELDS IN THE .txt MEAN
--------------------------------
The XML nests like:  Ableton > Groove > Clip > Value > MidiClip > ... > KeyTrack.
The parts that actually describe the groove:

  <Name Value="...">
      The groove/clip name (e.g. KAB1_137_AmenBreak_Cut_02).

  <Loop>
    <LoopStart Value="0"/>      Loop start, in BEATS.
    <LoopEnd   Value="4"/>      Loop end, in BEATS. This is the LENGTH of the
                                pattern. 4 = one bar of 4/4; 8 = two bars.
                                (HiddenLoopStart/End usually mirror these.)
    <LoopOn Value="true"/>      Whether looping is on.

  <Numerator   Value="4"/>      Time signature top  (beats per bar).
  <Denominator Value="4"/>      Time signature bottom (beat unit).
                                4/4 here.

  <FixedNumerator   Value="1"/> The editing/quantize grid: 1/16 means a
  <FixedDenominator Value="16"/> sixteenth-note grid. (Display only.)

  <KeyTrack> ... <MidiKey Value="36"/>
      Notes are grouped by pitch. MidiKey is the MIDI note number (0-127) that
      every MidiNoteEvent inside this KeyTrack plays.
      Reference: 60 = middle C (C4), so 36 = C2, 72 = C5. A drum groove is
      often all on one key (e.g. 36) because only the timing/velocity matter.

  <MidiNoteEvent Time="0.0131" Duration="0.0625" Velocity="127"
                 OffVelocity="64" IsEnabled="true"/>
      One hit / note:
        Time        = onset position in BEATS from the clip start. The
                      fractional part vs the grid is the GROOVE's microtiming
                      (e.g. 0.0131 = a hair after beat 1; 0.513 = just after
                      the "and" of beat 1). This is the swing/feel.
        Duration    = note length in BEATS (0.0625 = a 16th note).
        Velocity    = how hard/loud, 0-127. Often FRACTIONAL (e.g. 104.3) when
                      the groove was performed/humanized rather than programmed.
        OffVelocity = note-OFF (release) velocity, normally 64. Unused here.
        IsEnabled   = whether this note is active ("false" = muted).

  Everything else (LomId, LomIdView, WarpMarkers, MarkersGenerated,
  CurrentStart/End, ScrollerTimePreserver, etc.) is Ableton bookkeeping you can
  ignore for understanding the groove.


HOW THE APP USES IT
-------------------
Björk Cube reads Time + Velocity from each MidiNoteEvent and folds them onto a
16-step grid over the loop length (LoopEnd beats / 16 per slot):
  - per-slot TIMING offset (ms)  = how far each hit sits from the grid line
  - per-slot VELOCITY (0-1)      = Velocity / 127
That 16-slot timing+velocity profile is what the Groove instrument's bars show
and what the ".agr presets" load.
