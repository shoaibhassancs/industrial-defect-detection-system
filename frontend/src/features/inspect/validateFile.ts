/** Quick checks in the browser before anything is uploaded. The server repeats and enforces all of
 * them; these only save a round trip and give a clearer message. */

/** The backend's default limit (MAX_UPLOAD_MB). If it is changed there, the server's answer
 * (FILE_TOO_LARGE) is still shown correctly; this number only decides the early message. */
export const MAX_UPLOAD_MB = 10;

export type FileProblem = { code: "EMPTY" | "TOO_LARGE" | "NOT_JPEG_OR_PNG"; message: string };

const PNG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const JPEG = [0xff, 0xd8, 0xff];

function startsWith(bytes: Uint8Array, magic: number[]): boolean {
  return magic.every((b, i) => bytes[i] === b);
}

async function firstBytes(file: File, n: number): Promise<Uint8Array> {
  return new Uint8Array(await file.slice(0, n).arrayBuffer());
}

/** Returns a problem, or null if the file looks like a JPEG or PNG of acceptable size. The file
 * type is decided by its first bytes, not by its name or its declared type. */
export async function checkFile(file: File): Promise<FileProblem | null> {
  if (file.size === 0) return { code: "EMPTY", message: "That file is empty." };
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return {
      code: "TOO_LARGE",
      message: `That image is larger than ${MAX_UPLOAD_MB} MB. Choose a smaller one.`,
    };
  }
  const head = await firstBytes(file, 8);
  if (!startsWith(head, PNG) && !startsWith(head, JPEG)) {
    return { code: "NOT_JPEG_OR_PNG", message: "Choose a JPEG or PNG image (.jpg or .png)." };
  }
  return null;
}
