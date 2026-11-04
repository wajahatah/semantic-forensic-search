# Remote access to `lt-ai-server` from home: private mesh VPN (Tailscale)

**Date:** 2026-11-03 · **For:** Wajahat and the colleague who helps set it up · **Duration:** until the
experiments end; then revoke (checklist at the end). The office has no IT person, so we set this up
ourselves, on our own responsibility.

## Why this route

| Route | Exposed to the internet? | Needs admin rights on server / router? | Verdict |
|---|---|---|---|
| **Tailscale, user mode (this plan)** | **No.** The server only makes an outbound connection; no port is opened | No: it runs as our user | **Most secure option we can set up ourselves** |
| Router port-forward to SSH | Yes, one port open to the whole internet (bots scan it); restricting it to our home IP breaks when the home IP changes | Router admin + `sudo` for firewall/SSH settings | Not recommended |
| Public tunnel services (ngrok, Cloudflare quick tunnels) | Yes, a public URL | No | Not recommended |
| AnyDesk (already installed) | Already reachable by anyone who has its ID and password | — | Only a remote screen; our scripts need SSH |

With Tailscale, **only devices logged into our own Tailscale account** can reach the server, and only
the server itself, not the rest of the office network. SSH still needs our key on top of that.

**AnyDesk note:** AnyDesk is currently open to everyone, which is a separate risk on this machine.
Consider setting an unattended-access password and an allow-list in AnyDesk's security settings, or
turning off unattended access.

## Set-up (about 10 minutes; you run these, Claude does not)

Claude does not set up remote-access software or change security settings, so these steps are yours.
Nothing below needs `sudo`.

**1. Account:** create a free Tailscale account (tailscale.com), used only by you. Do not invite
other users to it.

**2. Server** (as `wajahat`, over SSH). Use the official static build:
```bash
mkdir -p ~/tailscale && cd ~/tailscale
curl -fsSLO https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz
tar xzf tailscale_latest_amd64.tgz --strip-components=1
nohup ./tailscaled --tun=userspace-networking --state=$HOME/tailscale/state \
      --socket=$HOME/tailscale/tailscaled.sock > $HOME/tailscale/tailscaled.log 2>&1 &
./tailscale --socket=$HOME/tailscale/tailscaled.sock up --hostname=lt-ai-server-sfs --accept-dns=false
```
The last command prints a login link. Open it in your browser and log in to your account. Then
`./tailscale --socket=$HOME/tailscale/tailscaled.sock ip -4` prints the server's private address
(100.x.y.z).

- **Do not** add `--advertise-routes` or `--advertise-exit-node`. They would open the office network
  or turn the server into an internet gateway.
- In user mode, incoming connections to the 100.x address are passed to the server's own ports,
  which is how SSH reaches it. We test this in step 4.

**3. Laptop:** install Tailscale for Windows from tailscale.com and log in to the same account. Add
to `C:\Users\LT\.ssh\config`:
```
Host lt_server_home
    HostName 100.x.y.z          # the address from step 2
    User wajahat
    ServerAliveInterval 30
```

**4. Test from the office first:** turn off the laptop's Wi-Fi, use a phone hotspot, and run
`ssh lt_server_home hostname`. It must print `lt-ai-server`.

**5. Lock it down** in the Tailscale admin console (login.tailscale.com):
- **Settings → Device approval:** on, so new devices need your approval.
- **Access controls:** allow only your laptop to reach the server on port 22:
  ```json
  {"acls": [{"action": "accept", "src": ["<your-laptop-name>"], "dst": ["lt-ai-server-sfs:22"]}]}
  ```
- **Machines:** check that only your laptop and `lt-ai-server-sfs` are listed.

**6. After a reboot:** the server is normally shut down at night, and `tailscaled` does not start by
itself. Ask the team to leave the server on during experiment nights. If it was restarted, re-run the
`nohup ./tailscaled ...` line from step 2 (no new login needed; the state is kept).

**Status 2026-11-03:** server joined (`lt-ai-server-sfs`, 100.79.95.104; checked: no routes, no exit node,
Tailscale SSH off). Laptop joined (100.89.102.18) and tested from another network: works. The server's host key over
Tailscale is the same as the trusted office key, so the SSH entry uses `HostKeyAlias 192.168.100.63`. Still to do:
device approval and the port-22 rule (step 5).

## While experiments run at night

- Jobs run **on the server**, under `nohup`, from `~/sfs/project`. The crop store was copied there
  over the office network, so the home connection only carries SSH and small result files.
- Jobs wait for free GPU memory before loading a model, so they do not push out the team's Ollama
  models.

## Revocation checklist (end of the experiments)

- [ ] Server: `~/tailscale/tailscale --socket=$HOME/tailscale/tailscaled.sock logout`, stop
      `tailscaled` (`kill <pid>`), `rm -rf ~/tailscale`
- [ ] Admin console: delete `lt-ai-server-sfs` (and the laptop if no longer needed)
- [ ] Laptop: log out of Tailscale or uninstall it; delete the `lt_server_home` SSH entry
- [ ] Server: remove our files (`~/sfs`, `~/envs/sfs`, `~/.config/sfs/secret`), check
      `ps -u wajahat` shows nothing of ours running
- [ ] Optional: remove the laptop's key from `~/.ssh/authorized_keys`
- [ ] Team: server back to its normal night shutdown
