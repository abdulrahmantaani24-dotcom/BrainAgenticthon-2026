# Agent-operated Colab control: verified limits and prepared execution

This workflow is for the agent, not an instruction for the user to run notebooks,
Python cells or upload bundles. The controller builds and uploads reviewed code
itself through the official Colab package once a secure connection is available.
It never downloads MRI, DICOM, clinical spreadsheets, private manifests, weights,
private reports or notebook outputs into the Codex machine.

## What was actually checked

The official repository is https://github.com/googlecolab/google-colab-cli .
Its source was reviewed at 5a119adc8e6be6f8b7aa04db57185aa6655a9171. The published
package google-colab-cli==0.7.5 installed successfully into a separate Python 3.12
virtual environment. The MRI training environment was not changed.

Unauthenticated HTTPS checks reached accounts.google.com (200), oauth2.googleapis.com
(token root GET 404), colab.research.google.com (200), colab.pa.googleapis.com (root
GET 404) and drive.google.com (200 after redirects). These establish network
reachability only, not sign-in, an API authorization grant, runtime allocation,
Drive access or GPU execution.

An actual official backend-listing attempt failed with DefaultCredentialsError.
No protected Google identity, ADC selector or identity manifest was configured in
the managed environment. No authenticated Colab command or remote code succeeded.
No VM was provisioned, no Drive data was read and no real training occurred.
The live auth failure must not be represented as successful empty session discovery.

## Precise authentication limitation

The published CLI's oauth2 provider uses Google's official remote authorization
landing page and **asks the operator to paste an authorization code**. This task
forbids transferring such codes in chat or logs, so the controller never invokes
that flow. It does not introduce an unofficial OAuth relay, collect a password,
request a refresh token, print credentials or use an insecure plaintext token cache.
It forces ADC, obtained only through an already configured protected credential
channel and Google's standard credential discovery/refresh implementation.

The CLI's own global scopes include Colab and Google identity scopes; Drive mount
permission is a separate Google consent operation. A generic Drive app connection
does not authenticate the Colab CLI. Network permissions alone cannot grant a
Google identity. The current available integrations have no Colab execution tool.

The alternative official https://github.com/googlecolab/colab-mcp requires the
agent client to run locally on the user's device and support dynamic tool-list
notifications. This cloud execution environment does not provide that browser
bridge or a secure OAuth callback UI. Therefore there is no verified one-click,
UI-only consent link that can complete authentication from this chat. A supported
Colab connection/credential channel must first be provided by the hosting client;
ordinary manual notebook operation is not a substitute for this missing capability.
No credentials or authorization codes should be pasted into chat or Git.

## Controller behavior

requirements-colab-control.txt pins the official package separately. The agent
uses scripts/colab_control.py with a private state directory outside Git. Operations:

- preflight: authenticated backend request only; never claims remote execution.
- probe: create an agent-owned T4 VM, execute a small Colab import/Python check,
  require its explicit success marker, and release only that VM.
- inventory: probe first, build the deterministic allowlisted bundle, upload **code
  only**, verify its SHA-256 and manifest remotely, install pinned dependencies,
  and execute execution.remote_job in a fresh remote Python process. The process
  avoids stale kernel imports. Only fixed, sanitized status fields are relayed.

The controller uses official command handlers with protected local state constructors.
This avoids the released CLI's hardcoded HOME log/settings writes in a read-only
cloud home; it does not alter HOME, authentication scopes, the provider or Google's
endpoints. This adapter is pinned and tested with mocks; remote provisioning,
WebSocket execution and Drive authorization have not been verified live here.

No notebook execution is sent through the CLI because the CLI normally saves its
output notebook locally. No remote download/log export operation is exposed.
Debug logging is suppressed; raw exceptions and remote stdout are not copied into
public logs. Session tokens belong only in mode-0700 private controller state with
mode-0600 files. Existing credentials are never printed or included in artifacts.
A runtime allocated by the controller is released in finally; existing unrelated
sessions are not adopted, stopped or modified. Quota/service failures remain blockers.

## Medical execution stays in private Colab

The only authorized data parent is
/content/drive/MyDrive/BrainMRI/datasets/DiseaseAcquisition/ . The remote job reads
only the four named dataset roots and explicitly configured metadata within that
parent. Reports/checkpoints/config snapshots are private under that authorized
parent, separately from all four source trees. The remote code does not mount Drive
interactively or bypass its consent; an unavailable mount/parent returns BLOCKED.
A newly allocated VM may require separate official Drive consent even after CLI
sign-in. No one-step solution is promised for that separate permission.

Source/license, clinical, label, modality, geometry, identity and patient-split gates
are unchanged. Supplied task configs must stay inside the authorized parent. Empty
configs produce blockers, never generated diagnoses or attestations. With explicit
execute-approved and an already approved config's real_training_enabled=true, the
remote job revalidates readiness, freezes unchanged reviewed data, runs a separate
one-batch smoke, then the configured training/resume and explicit final evaluation.
It never turns clinical approval flags on. An overfit experiment must have its own
reviewed private configuration; no actual overfit/training is asserted here.

Current four-source restrictions still block Task A registry eligibility and
non-BraTS Task B training. A connection alone would not approve these cohorts.
No medical model cards, learning curves, real checkpoints or clinical metrics can
be generated truthfully before an eligible real run. Existing research interfaces,
model inventory/report templates and Task C remain unchanged.

## Minimal intervention remaining

The missing prerequisite is a hosting-client-supported Google/Colab authentication
connection or a compatible local official MCP bridge. The user should only approve
Google's own sign-in/consent UI once that supported connection is available; the
agent must not request a secret or ask the user to operate notebooks. This cloud
chat currently lacks the channel needed to initiate/receive that UI-only consent.
The exact limitation is reported rather than inventing an authorization link.
