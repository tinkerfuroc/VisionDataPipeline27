import qrcode from "qrcode-generator";
import { useMemo, useState } from "react";

/** The address of the phone upload page for a project, as text and as a QR code to scan. */
export function PhoneLink({ projectId }: { projectId: string }) {
  const [open, setOpen] = useState(false);
  const url = `${window.location.origin}/upload?project=${encodeURIComponent(projectId)}`;
  const qr = useMemo(() => {
    const code = qrcode(0, "M");
    code.addData(url);
    code.make();
    return code.createDataURL(4, 8);
  }, [url]);
  const local = ["localhost", "127.0.0.1", "::1", "[::1]"].includes(window.location.hostname);
  if (!open) {
    return (
      <button type="button" className="btn" onClick={() => setOpen(true)}>
        Upload from a phone
      </button>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <p className="note">Scan this on the phone (same network), or open the address:</p>
      <img src={qr} alt="QR code for the phone upload page" className="w-40 self-center bg-white p-1" />
      <p className="break-all font-mono text-sm text-ink">{url}</p>
      {local && (
        <p className="note">
          This address only works on this computer. Open the network address printed by `vdp serve` on this computer
          instead, then use this button again.
        </p>
      )}
      <button type="button" className="btn" onClick={() => setOpen(false)}>
        Hide
      </button>
    </div>
  );
}
