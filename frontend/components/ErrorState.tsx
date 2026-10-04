export function ErrorState({ message }: { message: string }) {
  return (
    <p className="break-words rounded-md border border-amber-800 bg-amber-50 px-3 py-2 text-sm text-amber-950" role="alert">
      {message}
    </p>
  );
}
